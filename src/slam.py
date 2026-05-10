import os
import torch
import numpy as np
import time
from collections import OrderedDict
import torch.multiprocessing as mp

# Avoid `pidfd_getfd: Operation not permitted` under kernel.yama.ptrace_scope=1
# (default on recent Ubuntu kernels). PyTorch's default `file_descriptor` sharing
# strategy uses pidfd_getfd to receive shared-memory FDs from sibling processes,
# which the kernel rejects when ptrace_scope=1. `file_system` uses /dev/shm-backed
# files instead — slightly slower per share but no kernel-permission dependency.
# See https://github.com/pytorch/pytorch/issues/154566 for the upstream tracker.
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
        # ── Goal C: load precomputed external dynamic mask BEFORE forks ──
        # Numpy array lives in main process, fork-COW propagates to tracker
        # subprocess. The shared-memory `dynamic_masks` tensor handles the
        # tracker→BA hand-off. MUST run before `mp.Process(...)` calls in
        # run().
        self._maybe_load_semantic_masks(cfg, stream)

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
        n_dyn = float((masks < 0.5).mean()) * 100.0
        print(f"[INFO] semantic_mask: loaded {masks.shape} dtype={masks.dtype} "
              f"from {npz_path}  ({n_dyn:.2f}% dynamic).", flush=True)
        self.video.preload_dynamic_masks(masks)

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
