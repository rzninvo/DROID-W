import os
import torch
import numpy as np
import time
from collections import OrderedDict
import torch.multiprocessing as mp

# ── Fix `pidfd_getfd: Operation not permitted` under kernel.yama.ptrace_scope=1 ──
# Both CPU shared tensors AND CUDA shared tensors use pidfd_getfd in PyTorch 2.7
# to inherit memory-handle FDs from the parent. With Yama scope=1, the child
# (descendant) cannot ptrace the parent (ancestor) — only ancestors can ptrace
# descendants by default. The fix is two-pronged:
#   (a) PR_SET_PTRACER_ANY: parent declares any process may ptrace it, which
#       lets the child's pidfd_getfd succeed for CUDA tensor IPC reductions.
#   (b) file_system sharing strategy for CPU storages: avoids pidfd_getfd in
#       the CPU storage path entirely (uses /dev/shm files instead).
# Both apply at module import time, before any mp.Process is spawned.
# Refs: https://man7.org/linux/man-pages/man2/pidfd_getfd.2.html — permission
# governed by PTRACE_MODE_ATTACH_REALCREDS; pytorch/pytorch#154566 (upstream).
import ctypes
import ctypes.util as _ctypes_util
_PR_SET_PTRACER = 0x59616d61
# `(unsigned long)-1` per Linux kernel headers — our target is x86_64 / aarch64
# where c_ulong is 8 bytes. Fail loud if the assumption ever breaks (e.g. a
# 32-bit port) so we don't silently truncate the prctl arg. Reviewer A audit,
# Report 17 SHOULD-FIX #5.
assert ctypes.sizeof(ctypes.c_ulong) == 8, (
    f"slam.py: PR_SET_PTRACER_ANY assumes 64-bit c_ulong; this host has "
    f"{ctypes.sizeof(ctypes.c_ulong)}-byte c_ulong. Update _PR_SET_PTRACER_ANY "
    f"to (1 << 32) - 1 (or platform-appropriate) before continuing."
)
_PR_SET_PTRACER_ANY = (1 << 64) - 1
try:
    _libc = ctypes.CDLL(_ctypes_util.find_library("c"), use_errno=True)
    _libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                            ctypes.c_ulong, ctypes.c_ulong]
    _libc.prctl.restype = ctypes.c_int
    _ret = _libc.prctl(_PR_SET_PTRACER, _PR_SET_PTRACER_ANY, 0, 0, 0)
    if _ret != 0:
        _err = ctypes.get_errno()
        print(f"[WARN] slam.py: prctl(PR_SET_PTRACER_ANY) returned {_ret} "
              f"errno={_err} ({os.strerror(_err)}); CUDA mp tensors may fail "
              f"with pidfd_getfd EPERM under kernel.yama.ptrace_scope=1.",
              flush=True)
except OSError as _e:
    print(f"[WARN] slam.py: could not call prctl: {_e}; "
          f"CUDA mp tensors may fail with pidfd_getfd EPERM under "
          f"kernel.yama.ptrace_scope=1.", flush=True)
mp.set_sharing_strategy("file_system")

from munch import munchify

from src.modules.droid_net import DroidNet
from src.depth_video import DepthVideo
from src.trajectory_filler import PoseTrajectoryFiller
from src.utils.common import setup_seed, update_cam
from src.utils.Printer import Printer, FontColor
from src.utils.eval_traj import kf_traj_eval, full_traj_eval, full_traj_fill
from src.utils.datasets import BaseDataset
from src.tracker import Tracker
from src.mapper import Mapper
from src.backend import Backend
from src.utils.datasets import RGB_NoPose
from src.gui import gui_utils, slam_gui
from thirdparty.gaussian_splatting.scene.gaussian_model import GaussianModel
from torch.utils.tensorboard import SummaryWriter
from src.utils.sys_timer import timer
import ctypes

class SLAM:
    def __init__(self, cfg, stream: BaseDataset):
        super(SLAM, self).__init__()
        self.cfg = cfg
        self.device = cfg["device"]
        self.verbose: bool = cfg["verbose"]
        self.logger = None
        self.save_dir = cfg["data"]["output"] + "/" + cfg["scene"]

        os.makedirs(self.save_dir, exist_ok=True)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = update_cam(cfg)

        self.droid_net: DroidNet = DroidNet()

        self.printer = Printer(
            len(stream)
        )  # use an additional process for printing all the info

        self.load_pretrained(cfg)
        self.droid_net.to(self.device).eval()
        self.droid_net.share_memory()

        self.num_running_thread = torch.zeros((1)).int()
        self.num_running_thread.share_memory_()
        self.all_trigered = torch.zeros((1)).int()
        self.all_trigered.share_memory_()
        self.startup_barrier = None  # set in run() based on process count

        self.video = DepthVideo(cfg, self.printer)
        # ── Goal C: load precomputed external dynamic mask BEFORE process spawn ──
        # `run()` below uses `mp.set_start_method("spawn", force=True)`, so the
        # tracker child reconstructs SLAM via pickle (NOT fork-COW). The numpy
        # array `_frame_dynamic_masks_full` round-trips through pickle; the
        # GPU twin `_frame_dynamic_masks_full_t` rides CUDA IPC alongside the
        # other share_memory_()'d CUDA tensors. MUST run before any
        # mp.Process(...) calls in run() so both are populated pre-pickle.
        self._maybe_load_semantic_masks(cfg, stream)
        # ── Goal D.2: load precomputed radseg features BEFORE process spawn
        # (same spawn-pickle lifecycle as semantic_mask above). ──
        self._maybe_load_radseg_features(cfg, stream)

        self.ba = Backend(self.droid_net, self.video, self.cfg)

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(
            cfg=cfg,
            net=self.droid_net,
            video=self.video,
            printer=self.printer,
            device=self.device,
        )

        self.tracker: Tracker = None
        self.mapper: Mapper = None
        self.stream = stream
        self.final_clean = False

    def _maybe_load_semantic_masks(self, cfg, stream):
        """Load precomputed external dynamic mask from disk and hand to
        DepthVideo. Silent no-op when `tracking.semantic_mask.activate=False`.

        Search order for the mask file (per CLAUDE.md §6 — log explicitly
        what we did, no silent fallbacks):
          1) <data.input_folder>/<semantic_mask.path>
          2) <output>/<scene>/<semantic_mask.path>     (e.g. SLAM save_dir)
        """
        if not getattr(self.video, 'semantic_mask_aware', False):
            return
        sm_cfg = cfg.get('tracking', {}).get('semantic_mask', {})
        rel_path = sm_cfg.get('path', 'dynamic_masks.npz')
        from pathlib import Path as _Path
        candidates = []
        # Mirror BaseDataset's ROOT_FOLDER_PLACEHOLDER substitution
        # (datasets.py:110-112) so the path resolves the same way SLAM resolves
        # the dataset itself.
        input_folder = cfg.get('data', {}).get('input_folder', '')
        if input_folder and 'ROOT_FOLDER_PLACEHOLDER' in input_folder:
            input_folder = input_folder.replace(
                'ROOT_FOLDER_PLACEHOLDER', cfg['data'].get('root_folder', '.'))
        if input_folder:
            candidates.append(_Path(input_folder) / rel_path)
        save_dir = f"{cfg['data']['output']}/{cfg['scene']}"
        candidates.append(_Path(save_dir) / rel_path)
        npz_path = next((p for p in candidates if p.exists()), None)
        if npz_path is None:
            print(f"[WARN] semantic_mask.activate=True but mask file not found in any of "
                  f"{[str(p) for p in candidates]} — disabling semantic_mask "
                  f"(SLAM proceeds without external mask).", flush=True)
            self.video.semantic_mask_aware = False
            return
        masks_npz = np.load(str(npz_path))
        if 'mask' not in masks_npz.files:
            print(f"[WARN] semantic_mask: {npz_path} has no 'mask' field "
                  f"(found {masks_npz.files}) — disabling.", flush=True)
            self.video.semantic_mask_aware = False
            return
        masks = masks_npz['mask']  # (N_frames, H, W)
        if masks.ndim != 3:
            print(f"[WARN] semantic_mask: mask shape {masks.shape} is not 3D "
                  f"(N, H, W) — disabling.", flush=True)
            self.video.semantic_mask_aware = False
            return
        # Validate mask N matches stream length so a stale `dynamic_masks.npz`
        # from a different cut can't silently apply (Reviewer 2 audit, Report 17).
        # Tolerance of 1 covers the off-by-one from BaseDataset's edge crop.
        n_stream = len(stream)
        if abs(masks.shape[0] - n_stream) > 1:
            print(f"[WARN] semantic_mask: mask N={masks.shape[0]} disagrees with "
                  f"stream length {n_stream} by more than 1 — STALE FILE; refusing "
                  f"to apply (re-run scripts/precompute_dynamic_masks.py for this "
                  f"scene).", flush=True)
            self.video.semantic_mask_aware = False
            return
        n_dyn = float((masks < 0.5).mean()) * 100.0
        print(f"[INFO] semantic_mask: loaded {masks.shape} dtype={masks.dtype} "
              f"from {npz_path}  ({n_dyn:.2f}% dynamic).", flush=True)
        self.video.preload_dynamic_masks(masks)

    def _maybe_load_radseg_features(self, cfg, stream):
        """Load precomputed RADIO+SigLIP-2 features (Plan B v5 radseg_features.npz)
        and hand them to DepthVideo for D.2 weight composition. Silent no-op
        when tracking.semantic_weight.activate=False (or beta<=0, w_min>=1
        per plan-v2 Step 2's bypass switches in DepthVideo.__init__).

        Search order (per CLAUDE.md §6 — log what we did, no silent fallbacks):
          1) <data.input_folder>/<semantic_weight.path>
          2) <output>/<scene>/<semantic_weight.path>  (e.g. SLAM save_dir)
        """
        if not getattr(self.video, 'semantic_weight_aware', False):
            return
        sw_cfg = cfg.get('tracking', {}).get('semantic_weight', {})
        rel_path = sw_cfg.get('path', 'radseg_features.npz')
        from pathlib import Path as _Path
        candidates = []
        input_folder = cfg.get('data', {}).get('input_folder', '')
        if input_folder and 'ROOT_FOLDER_PLACEHOLDER' in input_folder:
            input_folder = input_folder.replace(
                'ROOT_FOLDER_PLACEHOLDER', cfg['data'].get('root_folder', '.'))
        if input_folder:
            candidates.append(_Path(input_folder) / rel_path)
        save_dir = f"{cfg['data']['output']}/{cfg['scene']}"
        candidates.append(_Path(save_dir) / rel_path)
        npz_path = next((p for p in candidates if p.exists()), None)
        if npz_path is None:
            print(f"[WARN] semantic_weight.activate=True but radseg feature file "
                  f"not found in any of {[str(p) for p in candidates]} — "
                  f"disabling D.2 (SLAM proceeds without semantic weight).",
                  flush=True)
            self.video.semantic_weight_aware = False
            return
        feats_npz = np.load(str(npz_path))
        for field in ('lang_aligned_feats', 'kf_indices'):
            if field not in feats_npz.files:
                print(f"[WARN] semantic_weight: {npz_path} missing '{field}' "
                      f"(found {feats_npz.files}) — disabling D.2.", flush=True)
                self.video.semantic_weight_aware = False
                return
        feats = feats_npz['lang_aligned_feats']
        # ── Plan-v2 §Step 3a Option A: use kf_global_indices (dataset frame
        # index per row), NOT kf_indices (local KF position [0..N-1]). The
        # old (schema v1) consumer path looked up frame_idx in a dict keyed by
        # local position → mostly-wrong row or miss. Reject v1 files unless
        # the legacy_override flag is set; otherwise build the mapping from
        # the v2 kf_global_indices field. ──
        schema_version = int(feats_npz.get('schema_version', np.int64(1)))
        legacy_override = bool(sw_cfg.get('legacy_override_v1', False))
        if 'kf_global_indices' not in feats_npz.files:
            if legacy_override:
                print(f"[WARN] semantic_weight: {npz_path} is schema v1 (no "
                      f"kf_global_indices); legacy_override_v1=True so falling "
                      f"back to kf_indices (identity). Lookup will be wrong "
                      f"for any KF whose frame_idx != local position.",
                      flush=True)
                kf_global_indices = feats_npz['kf_indices'].astype(np.int64)
            else:
                print(f"[WARN] semantic_weight: {npz_path} lacks "
                      f"'kf_global_indices' (schema v1, Plan-v2 §Step 3a "
                      f"identified this as the source of D.2's mostly-wrong "
                      f"feature lookup). Re-run scripts/precompute_radseg_features.py "
                      f"or set tracking.semantic_weight.legacy_override_v1=True "
                      f"to use the broken v1 contract anyway. Disabling D.2.",
                      flush=True)
                self.video.semantic_weight_aware = False
                return
        else:
            kf_global_indices = feats_npz['kf_global_indices'].astype(np.int64)
            # Detect the v1-but-saved-as-v2 case (legacy precompute saved
            # kf_global_indices = kf_indices). If global == local AND not a
            # match against any plausible frame-index pattern, warn loudly.
            kf_local = feats_npz['kf_indices'].astype(np.int64)
            if (schema_version < 2 and
                    np.array_equal(kf_global_indices, kf_local) and
                    not legacy_override):
                print(f"[WARN] semantic_weight: {npz_path} stores "
                      f"kf_global_indices == kf_indices (legacy identity); "
                      f"schema_version={schema_version} < 2. Re-precompute "
                      f"to get true frame indices, or set legacy_override_v1=True. "
                      f"Disabling D.2 to avoid the silent bug.", flush=True)
                self.video.semantic_weight_aware = False
                return
        radio_version = str(feats_npz.get('radio_version', np.array('unknown')))
        lang_adaptor = str(feats_npz.get('lang_adaptor', np.array('unknown')))
        # PCA-256 compression cuts the per-KF buffer by 6x (1536 -> 256 dims).
        # When pca_basis is set in cfg, load it and pass to preload; otherwise
        # store raw and pay the memory bill (K must equal D_raw).
        pca_basis_rel = sw_cfg.get('pca_basis', None)
        pca_mean = None
        pca_components = None
        if pca_basis_rel:
            from pathlib import Path as _Path
            import torch as _torch
            pca_path = _Path(pca_basis_rel)
            if not pca_path.is_absolute():
                # Repo-root-relative (matches scripts/query_panoptic.py convention).
                pca_path = _Path(__file__).resolve().parents[1] / pca_basis_rel
            if not pca_path.exists():
                print(f"[WARN] semantic_weight: pca_basis={pca_basis_rel} not "
                      f"found at {pca_path} — disabling D.2.", flush=True)
                self.video.semantic_weight_aware = False
                return
            state = _torch.load(str(pca_path), map_location='cpu', weights_only=False)
            pca_mean = state['mean']
            pca_components = state['components']
            print(f"[INFO] semantic_weight: loaded pca_basis from {pca_path} "
                  f"(D={state['feature_dim']} -> K={state['target_dim']}, "
                  f"var_explained={state.get('fit_variance_explained', 'n/a')}).",
                  flush=True)
        print(f"[INFO] semantic_weight: loaded radseg_features from {npz_path}; "
              f"{feats.shape} dtype={feats.dtype}, radio={radio_version}, "
              f"adaptor={lang_adaptor}, schema_version={schema_version}, "
              f"N_kf_precomp={len(kf_global_indices)}.", flush=True)
        # First few mappings — would have caught the v1 bug immediately.
        head = kf_global_indices[:5].tolist()
        print(f"[INFO] semantic_weight: row -> frame_idx mapping (first 5): "
              f"{list(enumerate(head))}", flush=True)
        self.video.preload_radseg_features(kf_global_indices, feats,
                                           pca_mean=pca_mean,
                                           pca_components=pca_components)

    def load_pretrained(self, cfg):
        droid_pretrained = cfg["tracking"]["pretrained"]
        state_dict = OrderedDict(
            [
                (k.replace("module.", ""), v)
                for (k, v) in torch.load(droid_pretrained, weights_only=True).items()
            ]
        )
        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]
        self.droid_net.load_state_dict(state_dict)
        self.droid_net.eval()
        self.printer.print(
            f"Load droid pretrained checkpoint from {droid_pretrained}!", FontColor.INFO
        )

    def tracking(self, pipe):
        # clean all event writer files
        for file in os.listdir(self.save_dir):
            if file.startswith("events.out.tfevents."):
                os.remove(os.path.join(self.save_dir, file))
                
        event_writer = SummaryWriter(self.save_dir)
        self.tracker = Tracker(self, pipe, event_writer)
        self.printer.print("Tracking Triggered!", FontColor.TRACKER)
        self.all_trigered += 1

        os.makedirs(f"{self.save_dir}/mono_priors/depths", exist_ok=True)
        os.makedirs(f"{self.save_dir}/mono_priors/features", exist_ok=True)

        if self.startup_barrier is not None:
            self.startup_barrier.wait()
        self.printer.pbar_ready()
        self.tracker.run(self.stream)
        self.printer.print("Tracking Done!", FontColor.TRACKER)

        if not self.cfg["mapping"]["enable"]:
            self.terminate()

    def mapping(self, pipe, q_main2vis, q_vis2main):
        self.mapper = Mapper(self, pipe, q_main2vis, q_vis2main)
        self.printer.print("Mapping Triggered!", FontColor.MAPPER)

        self.all_trigered += 1
        setup_seed(self.cfg["setup_seed"])

        if self.startup_barrier is not None:
            self.startup_barrier.wait()
        self.mapper.run()
        self.printer.print("Mapping Done!", FontColor.MAPPER)

        if self.cfg["mapping"]["enable"]:
            self.terminate()

    @timer.section("Final Global BA")
    def backend(self):
        self.printer.print("Final Global BA Triggered!", FontColor.TRACKER)

        metric_depth_reg_activated = self.video.metric_depth_reg
        if metric_depth_reg_activated:
            self.video.metric_depth_reg = False

        self.ba = Backend(self.droid_net, self.video, self.cfg)
        self.ba.dense_ba(7, enable_udba=self.cfg['tracking']['frontend']['enable_opt_dyn_mask'])
        self.ba.dense_ba(12, enable_udba=self.cfg['tracking']['frontend']['enable_opt_dyn_mask'], save_edges_weights=False)
        self.printer.print("Final Global BA Done!", FontColor.TRACKER)

        if metric_depth_reg_activated:
            self.video.metric_depth_reg = True

    def terminate(self):
        """fill poses for non-keyframe images and evaluate"""

        if (
            self.cfg["tracking"]["backend"]["final_ba"]
            and self.cfg["mapping"]["eval_before_final_ba"]
        ):
            self.video.save_video(f"{self.save_dir}/video.npz")
            if not isinstance(self.stream, RGB_NoPose):
                try:
                    ate_statistics, global_scale, r_a, t_a = kf_traj_eval(
                        f"{self.save_dir}/video.npz",
                        f"{self.save_dir}/traj/before_final_ba",
                        "kf_traj",
                        self.stream,
                        self.logger,
                        self.printer,
                    )
                except Exception as e:
                    self.printer.print(e, FontColor.ERROR)
            if self.cfg["mapping"]["enable"]:
                self.mapper.save_all_kf_figs(
                    self.save_dir,
                    iteration="before_refine",
                )
            if self.cfg["tracking"]["uncertainty_params"]["visualize"]:
                self.video.visualize_all_opt_params(
                    self.save_dir,
                    iteration="final",
                )

        if self.cfg["tracking"]["backend"]["final_ba"]:
            self.backend()

        self.video.save_video(f"{self.save_dir}/video.npz")
        if not isinstance(self.stream, RGB_NoPose):
            try:
                ate_statistics, global_scale, r_a, t_a = kf_traj_eval(
                    f"{self.save_dir}/video.npz",
                    f"{self.save_dir}/traj",
                    "kf_traj",
                    self.stream,
                    self.logger,
                    self.printer,
                )
            except Exception as e:
                self.printer.print(e, FontColor.ERROR)

        if self.cfg["mapping"]["enable"]:
            if self.cfg["tracking"]["backend"]["final_ba"]:
                self.mapper.final_refine(
                    iters=self.cfg["mapping"]["final_refine_iters"]
                )  # this performs a set of optimizations with RGBD loss to correct

            # Evaluate the metrics
            self.mapper.save_all_kf_figs(
                self.save_dir,
                iteration="after_refine",
            )

            # Regenerate feature extractor for non-keyframes
            self.traj_filler.setup_feature_extractor()
            traj_est = full_traj_fill(
                self.traj_filler,
                self.mapper,
                self.stream,
                fast_mode=self.cfg['fast_mode'],
            )
            full_traj_eval(traj_est, self.stream, self.printer, self.logger, f"{self.save_dir}/traj", "full_traj")
            
            self.mapper.gaussians.save_ply(f"{self.save_dir}/final_gs.ply")
            
        else:
            traj_est = None
            with timer.section("Full Trajectory Filling"):
                self.traj_filler.setup_feature_extractor()
                traj_est = full_traj_fill(
                    self.traj_filler,
                    None,
                    self.stream,
                    fast_mode=True,
                )
            full_traj_eval(traj_est, self.stream, self.printer, self.logger, f"{self.save_dir}/traj", "full_traj")

        self.printer.print("Metrics Evaluation Done!", FontColor.EVAL)
        timer._report_summary(self.save_dir)
        self.final_clean = True

    def _eval_depth_all(self, ate_statistics, global_scale, r_a, t_a):
        """From Splat-SLAM. Not used in WildGS-SLAM evaluation, but might be useful in the future."""
        # Evaluate depth error
        self.printer.print(
            "Evaluate sensor depth error with per frame alignment", FontColor.EVAL
        )
        depth_l1, depth_l1_max_4m, coverage = self.video.eval_depth_l1(
            f"{self.save_dir}/video.npz", self.stream
        )
        self.printer.print("Depth L1: " + str(depth_l1), FontColor.EVAL)
        self.printer.print("Depth L1 mask 4m: " + str(depth_l1_max_4m), FontColor.EVAL)
        self.printer.print("Average frame coverage: " + str(coverage), FontColor.EVAL)

        self.printer.print(
            "Evaluate sensor depth error with global alignment", FontColor.EVAL
        )
        depth_l1_g, depth_l1_max_4m_g, _ = self.video.eval_depth_l1(
            f"{self.save_dir}/video.npz", self.stream, global_scale
        )
        self.printer.print("Depth L1: " + str(depth_l1_g), FontColor.EVAL)
        self.printer.print(
            "Depth L1 mask 4m: " + str(depth_l1_max_4m_g), FontColor.EVAL
        )

        # save output data to dict
        # File path where you want to save the .txt file
        file_path = f"{self.save_dir}/depth_stats.txt"
        integers = {
            "depth_l1": depth_l1,
            "depth_l1_global_scale": depth_l1_g,
            "depth_l1_mask_4m": depth_l1_max_4m,
            "depth_l1_mask_4m_global_scale": depth_l1_max_4m_g,
            "Average frame coverage": coverage,  # How much of each frame uses depth from droid (the rest from Omnidata)
            "traj scaling": global_scale,
            "traj rotation": r_a,
            "traj translation": t_a,
            "traj stats": ate_statistics,
        }
        # Write to the file
        with open(file_path, "w") as file:
            for label, number in integers.items():
                file.write(f"{label}: {number}\n")

        self.printer.print(f"File saved as {file_path}", FontColor.EVAL)

    def run(self):
        mp.set_start_method("spawn", force=True)
        exit_event = mp.Event()

        m_pipe, t_pipe = mp.Pipe()

        q_main2vis = mp.Queue() if self.cfg['gui'] else None
        q_vis2main = mp.Queue() if self.cfg['gui'] else None

        if self.cfg['mapping']['enable']:
            processes = [
                mp.Process(target=self.tracking, args=(t_pipe,)),                       # call tracking() function
                mp.Process(target=self.mapping, args=(m_pipe,q_main2vis,q_vis2main)),   # call mapping() function
            ]
        else:
            processes = [
                mp.Process(target=self.tracking, args=(t_pipe,)),                       # call tracking() function
            ]
        self.num_running_thread[0] += len(processes)
        self.startup_barrier = mp.Barrier(len(processes))
        for p in processes:
            p.start()

        if self.cfg['gui']:
            time.sleep(5)
            pipeline_params = munchify(self.cfg["mapping"]["pipeline_params"])
            bg_color = [0, 0, 0]
            background = torch.tensor(
                bg_color, dtype=torch.float32, device=self.device
            )
            gaussians = GaussianModel(self.cfg['mapping']['model_params']['sh_degree'], config=self.cfg)

            params_gui = gui_utils.ParamsGUI(
                pipe=pipeline_params,
                background=background,
                gaussians=gaussians,
                q_main2vis=q_main2vis,
                q_vis2main=q_vis2main,
            )
            gui_process = mp.Process(target=slam_gui.run, args=(params_gui,))
            gui_process.start()
            # self.num_running_thread[0] += 1

        # visualizer
        if self.cfg['droidvis']:
            from src.utils.droid_visualization_rerun import droid_visualization_rerun
            self.visualizer = mp.Process(
                target=droid_visualization_rerun,
                args=(self.video,),
                kwargs=dict(
                    web_port=9876,                           # port the node will serve on
                    record_path=f"{self.save_dir}/rerun_stream.rrd",  # optional
                    exit_event=exit_event,
                )
            )
            self.visualizer.start()


        for p in processes:
            p.join()
        
        # detect if the visualizer is still running
        if self.cfg['droidvis'] and self.visualizer.is_alive():
            exit_event.set()
            self.visualizer.join(timeout=10)
        
        self.printer.terminate()

        for process in mp.active_children():
            process.terminate()
            process.join()

def gen_pose_matrix(R, T):
    pose = np.eye(4)
    pose[0:3, 0:3] = R.cpu().numpy()
    pose[0:3, 3] = T.cpu().numpy()
    return pose
