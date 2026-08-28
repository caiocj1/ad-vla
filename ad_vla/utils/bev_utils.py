import torch
from simple_bev.utils import geom, basic

from ad_vla.dataset.data_types import CalibrationDict


def build_simple_bev_calibs(
    calib_list: list[CalibrationDict],
    batch_size=1,
    image_hw=None,  # (H, W) of the actual images fed to the model
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Transform canonical camera calibrations into SimpleBEV matrices.

    The input calibrations must already use the repository's normalized
    optical/RDF camera convention: camera +x is image-right, +y is image-down,
    and +z points forward through the lens. Dataset loaders are responsible for
    converting legacy FLU cameras before they reach this helper; ``is_flu=True``
    is rejected here so BEV targets cannot silently mix frame conventions.

    ``translation`` and ``rotation`` are interpreted as the camera-to-ego
    transform from ``CalibrationDict``. Intrinsics are scaled from
    ``original_img_size`` to ``image_hw`` before being packed into the
    ``pix_T_cams`` matrices expected by SimpleBEV.

    Args:
        calib_list (list[CalibrationDict]): calibrations of cameras
            from a single timestep. Length S = number of input cameras.

    Returns:
      pix_T_cams:   [B, S, 4, 4]
      cam0_T_camXs: [B, S, 4, 4]
    """
    if image_hw is None:
        raise ValueError("image_hw must be provided, e.g. (448, 800)")

    H, W = image_hw

    Ks = []
    Rs = []
    ts = []

    for c in calib_list:
        if c.is_flu:
            raise ValueError(
                "build_simple_bev_calibs expects optical/RDF calibrations "
                "(is_flu=False). Rebuild legacy FLU samples with the normalized "
                "dataset loaders before using BEV targets."
            )

        if c.intrinsics is None:
            raise ValueError("Each camera needs intrinsics")

        K = c.intrinsics.clone().float()

        # scale intrinsics from original image size to current image size
        H0, W0 = c.original_img_size
        sx = W / W0
        sy = H / H0
        K = geom.scale_intrinsics(K.unsqueeze(0), sx, sy).squeeze(0)

        # SimpleBEV expects optical/RDF camera coordinates for pinhole
        # projection, which is the normalized saved calibration convention.
        R = c.rotation.float()
        t = c.translation.float()

        Ks.append(K)
        Rs.append(R)
        ts.append(t)

    intrins = (
        torch.stack(Ks, dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    )  # [B,S,3,3]
    rots = torch.stack(Rs, dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [B,S,3,3]
    trans = torch.stack(ts, dim=0).unsqueeze(0).repeat(batch_size, 1, 1)  # [B,S,3]

    B = intrins.shape[0]
    __p = lambda x: basic.pack_seqdim(x, B)
    __u = lambda x: basic.unpack_seqdim(x, B)

    pix_T_cams_ = geom.merge_intrinsics(*geom.split_intrinsics(__p(intrins)))
    pix_T_cams = __u(pix_T_cams_)  # [B,S,4,4]

    ego_T_camXs = geom.merge_rtlist(rots, trans)  # [B,S,4,4]
    cam0_T_camXs = geom.get_camM_T_camXs(ego_T_camXs, ind=0)

    return pix_T_cams, cam0_T_camXs
