import torch
from simple_bev import saverloader
from simple_bev.utils import vox
from simple_bev.nets.segnet import Segnet

from ad_vla.dataset.data_types import E2EDataBatch, TrajectorySampling
from ad_vla.utils.bev_utils import build_simple_bev_calibs
from ad_vla.utils.traj_utils import resample_tensor


class BEVReward:
    def __init__(
        self,
        bev_model_ckpt_path: str,
    ):
        # Initialize pre-trained SimpleBEV model
        scene_centroid = torch.tensor([[0.0, 1.0, 0.0]])
        bounds = (-50, 50, -5, 5, -50, 50)  # XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX
        Z, Y, X = 200, 8, 200

        self.vox_util = vox.Vox_util(
            Z,
            Y,
            X,
            scene_centroid=scene_centroid,
            bounds=bounds,
            assert_cube=False,
        )

        self.bev_model = Segnet(
            Z,
            Y,
            X,
            self.vox_util,
            use_radar=False,
            use_lidar=False,
            use_metaradar=False,
            do_rgbcompress=True,
            encoder_type="res101",
        )
        self.bev_model.requires_grad_(False)

        saverloader.load(bev_model_ckpt_path, self.bev_model)
        self.bev_model.eval()

    def __call__(self, pred_trajs: torch.Tensor, batch: E2EDataBatch) -> torch.Tensor:
        """
        Computes BEV reward for predicted trajectories.

        Args:
            pred_trajs (torch.Tensor): Predicted trajectories shaped [B, G, T, 2].
            batch (E2EDataBatch): Data batch with ground truth trajectories.

        Returns:
            torch.Tensor: Rewards shaped [B, G]

        B = Batch size, G = Group size (number of predictions per sample), T = Trajectory length
        """
        gt_trajs = torch.stack([sample.fut_traj for sample in batch.samples]).to(
            pred_trajs.device
        )
        gt_traj_sampling = batch.samples[0].fut_traj_sampling

        # Resample to fine-grained
        target_sampling = TrajectorySampling(
            time_horizon=gt_traj_sampling.time_horizon, interval_length=0.1
        )

        if target_sampling != gt_traj_sampling:
            gt_trajs = resample_tensor(gt_trajs, gt_traj_sampling, target_sampling)

        # Calculate L2 reward
        l2_reward = -(pred_trajs - gt_trajs[:, None]).norm(dim=-1).mean(dim=-1)

        # Get coordinate positions in BEV grid
        quant_pred_trajs = torch.cat(
            [
                (2 * pred_trajs[..., 0:1]).int(),
                (50 - 2 * pred_trajs[..., 1:2]).int(),
            ],
            dim=-1,
        )
        bev_occ_map = self.get_bev_occ_map(batch).to(pred_trajs.device)

        # Save mask with valid positions
        bev_valid_mask = (quant_pred_trajs < 100) & (quant_pred_trajs >= 0)
        quant_pred_trajs[~bev_valid_mask] = 0

        batch_size = bev_occ_map.shape[0]

        batch_idx = torch.arange(batch_size, device=bev_occ_map.device)[:, None, None]

        # Take values along trajectory
        # BEV values are in the range of 0 to 1, indicating absence/presence of obstacle
        bev_values = -bev_occ_map[:, 100:, 50:150][
            batch_idx,
            quant_pred_trajs[..., 0],
            quant_pred_trajs[..., 1],
        ].sigmoid()
        bev_reward = (bev_values * bev_valid_mask.all(dim=-1)).sum(dim=-1) / (
            bev_valid_mask.all(dim=-1).sum(dim=-1) + 1e-6
        )

        reward = bev_reward + l2_reward

        return reward

    def get_bev_occ_map(self, batch: E2EDataBatch) -> torch.Tensor:
        # Prefer privileged teacher cameras when datasets provide them. These
        # are fixed-size last-timestep images with matching calibrations.
        camera_frames = [
            (
                sample.teacher_cameras
                if sample.teacher_cameras is not None
                else sample.cameras
            )[-1]
            for sample in batch.samples
        ]

        # Retrieve available cameras
        key_list = list(camera_frames[0].keys())
        key_list.remove("FRONT")
        key_list.insert(0, "FRONT")

        pix_T_cams = []
        cam0_T_camXs = []
        for camera_frame in camera_frames:
            sample_h, sample_w, _ = camera_frame["FRONT"].image.shape
            cal_list = [camera_frame[k].cal_dict for k in key_list]
            pix_T_cam, cam0_T_camX = build_simple_bev_calibs(
                cal_list, image_hw=(sample_h, sample_w)
            )
            pix_T_cams.append(pix_T_cam)
            cam0_T_camXs.append(cam0_T_camX)
        pix_T_cams = torch.cat(pix_T_cams, dim=0)
        cam0_T_camXs = torch.cat(cam0_T_camXs, dim=0)

        rgb_imgs = [
            torch.stack([camera_frame[k].image for k in key_list])
            for camera_frame in camera_frames
        ]
        rgb_imgs = torch.stack(rgb_imgs)
        rgb_imgs = rgb_imgs.permute(0, 1, 4, 2, 3).float() / 255.0
        rgb_imgs = rgb_imgs - 0.5

        self.bev_model.cuda()
        self.bev_model.eval()
        device = next(self.bev_model.parameters()).device
        with torch.inference_mode():
            bev_out = self.bev_model(
                rgb_camXs=rgb_imgs.to(device),  # [B,S,3,H,W]
                pix_T_cams=pix_T_cams.to(device),  # [B,S,4,4]
                cam0_T_camXs=cam0_T_camXs.to(device),  # [B,S,4,4]
                vox_util=self.vox_util,
                rad_occ_mem0=None,  # camera-only model
            )
        self.bev_model.cpu()
        torch.cuda.empty_cache()

        bev_seg_map = bev_out[2]
        if bev_seg_map.ndim == 4 and bev_seg_map.shape[1] == 1:
            bev_seg_map = bev_seg_map[:, 0]
        return bev_seg_map
