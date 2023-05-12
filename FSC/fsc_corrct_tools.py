
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from typing import List, Sequence, Optional, Tuple, Callable, Union
from numpy.typing import ArrayLike, NDArray, DTypeLike
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes


NDArrayInt = NDArray[np.signedinteger]

def circular_mask(
    vol_shape_zxy: Union[Sequence[int], NDArrayInt],
    radius_offset: float = 0,
    coords_ball: Union[Sequence[int], NDArrayInt, None] = None,
    vol_origin_zxy: Optional[Sequence[float]] = None,
    taper_func: Optional[str] = None,
    taper_target: str = "edge",
    super_sampling: int = 1,
    dtype: DTypeLike = np.float32,
) -> NDArray:
    """
    Compute a circular mask for the reconstruction volume.
    Parameters
    ----------
    vol_shape_zxy : Sequence[int] | NDArrayInt
        The size of the volume.
    radius_offset : float, optional
        The offset with respect to the volume edge. The default is 0.
    coords_ball : Sequence[int] | NDArrayInt | None, optional
        The coordinates to consider for the non-masked region. The default is None.
    vol_origin_zxy : Optional[Sequence[float]], optional
        The origin of the coordinates in voxels. The default is None.
    taper_func : str, optional
        The mask data type. Allowed types: "const" | "cos". The default is "const".
    super_sampling : int, optional
        The pixel super sampling to be used for the mask. The default is 1.
    dtype : DTypeLike, optional
        The type of mask. The default is np.float32.
    Raises
    ------
    ValueError
        In case of unknown taper_func value, or mismatching volume origin and shape.
    Returns
    -------
    NDArray
        The circular mask.
    """
    vol_shape_zxy_s = np.array(vol_shape_zxy, dtype=int) * super_sampling

    coords = [
        np.linspace(-(s - 1) / (2 * super_sampling), (s - 1) / (2 * super_sampling), s, dtype=dtype) for s in vol_shape_zxy_s
    ]
    if vol_origin_zxy:
        if len(coords) != len(vol_origin_zxy):
            raise ValueError(f"The volume shape ({len(coords)}), and the origin shape ({len(vol_origin_zxy)}) should match")
        coords = [c + vol_origin_zxy[ii] for ii, c in enumerate(coords)]
    coords = np.meshgrid(*coords, indexing="ij")

    if coords_ball is None:
        coords_ball = np.arange(-np.fmin(2, len(vol_shape_zxy_s)), 0, dtype=int)
    else:
        coords_ball = np.array(coords_ball, dtype=int)

    max_radius = np.min(vol_shape_zxy_s[coords_ball]) / (2 * super_sampling) + radius_offset

    coords = np.stack(coords, axis=0)
    if coords_ball.size == 1:
        dists = np.abs(coords[coords_ball, ...])
    else:
        dists = np.sqrt(np.sum(coords[coords_ball, ...] ** 2, axis=0))

    if taper_func is None:
        mask = (dists <= max_radius).astype(dtype)
    elif isinstance(taper_func, str):
        if taper_target.lower() == "edge":
            cut_off_denom = 2
        elif taper_target.lower() == "diagonal":
            cut_off_denom = np.sqrt(2)
        else:
            raise ValueError(
                f"Parameter `taper_target` should be one of: 'edge' or 'diagonal', but {taper_target} passed instead."
            )

        if taper_func.lower() == "cos":
            cut_off_radius = np.min(vol_shape_zxy_s[coords_ball]) / (cut_off_denom * super_sampling)
            cut_off_size = cut_off_radius - max_radius
            outter_vals = np.cos(np.fmax(dists - max_radius, 0) / cut_off_size * np.pi) / 2 + 0.5
            mask = (outter_vals * (dists < cut_off_radius)).astype(dtype)
        else:
            raise ValueError(f"Unknown taper function: {taper_func}")
    else:
        raise ValueError(f"Parameter `taper_func` should either be a string or None.")

    if super_sampling > 1:
        new_shape = np.stack([np.array(vol_shape_zxy), np.ones_like(vol_shape_zxy) * super_sampling], axis=1).flatten()
        mask = mask.reshape(new_shape)
        mask = np.mean(mask, axis=tuple(np.arange(1, len(vol_shape_zxy) * 2, 2, dtype=int)))

    return mask


def ball(
    data_shape_vu: ArrayLike,
    radius: Union[int, float],
    super_sampling: int = 5,
    dtype: DTypeLike = np.float32,
    func: Optional[Callable] = None,
) -> ArrayLike:
    """
    Compute a ball with specified radius.
    Parameters
    ----------
    data_shape_vu : ArrayLike
        Shape of the output array.
    radius : int | float
        Radius of the ball.
    super_sampling : int, optional
        Super-sampling for having smoother ball edges. The default is 5.
    dtype : DTypeLike, optional
        Type of the output. The default is np.float32.
    func : Optional[Callable], optional
        Point-wise function for the local values. The default is None.
    Returns
    -------
    ArrayLike
        The ball.
    """
    data_shape_vu = np.array(data_shape_vu, dtype=int) * super_sampling

    # coords = [np.linspace(-(s - 1) / 2, (s - 1) / 2, s, dtype=np.float32) for s in data_shape_vu]
    coords = [np.fft.fftfreq(d, 1 / d) for d in data_shape_vu]
    coords = np.stack(np.meshgrid(*coords, indexing="ij"), axis=0)

    r = np.sqrt(np.sum(coords**2, axis=0)) / super_sampling

    probe = (r < radius).astype(dtype)
    if func is not None:
        probe *= func(r)

    probe = np.roll(probe, super_sampling // 2, axis=tuple(np.arange(len(data_shape_vu))))
    new_shape = np.stack([data_shape_vu // super_sampling, np.ones_like(data_shape_vu) * super_sampling], axis=1).flatten()
    probe = probe.reshape(new_shape)
    probe = np.mean(probe, axis=tuple(np.arange(1, len(data_shape_vu) * 2, 2, dtype=int)))

    return np.fft.fftshift(probe)


def azimuthal_integration(img: NDArray, axes: Sequence[int] = (-2, -1), domain: str = "direct") -> NDArray:
    """
    Compute the azimuthal integration of a n-dimensional image or a stack of them.
    Parameters
    ----------
    img : NDArray
        The image or stack of images.
    axes : tuple(int, int), optional
        Axes of that need to be azimuthally integrated. The default is (-2, -1).
    domain : string, optional
        Domain of the integration. Options are: "direct" | "fourier". Default is "direct".
    Raises
    ------
    ValueError
        Error returned when not passing images or wrong axes.
    Returns
    -------
    NDArray
        The azimuthally integrated profile.
    """
    num_dims_int = len(axes)
    num_dims_img = len(img.shape)

    if num_dims_img < num_dims_int:
        raise ValueError(
            "Input image ({num_dims_img}D) should be at least the same dimensionality"
            " of the axes for the integration (#{num_dims_int})."
        )
    if len(axes) == 0:
        raise ValueError("Input axes should be at least 1.")

    # Compute the coordinates of the pixels along the chosen axes
    img_axes_dims = np.array(np.array(img.shape)[list(axes)], ndmin=1)
    if domain.lower() == "direct":
        half_dims = (img_axes_dims - 1) / 2
        coords = [np.linspace(-h, h, d) for h, d in zip(half_dims, img_axes_dims)]
    else:
        coords = [np.fft.fftfreq(d, 1 / d) for d in img_axes_dims]
    coords = np.stack(np.meshgrid(*coords, indexing="ij"))
    r = np.sqrt(np.sum(coords**2, axis=0))

    # Reshape the volume to have the axes to be integrates as right-most axes
    img_tr_op = np.array([*range(len(img.shape))])
    img_tr_op = np.concatenate((np.delete(img_tr_op, obj=axes), img_tr_op[list(axes)]))
    img = np.transpose(img, img_tr_op)

    if num_dims_img > num_dims_int:
        img_old_shape = img.shape[:-num_dims_int]
        img = np.reshape(img, [-1, *img_axes_dims])

    # Compute the linear interpolation coefficients
    r_l = np.floor(r)
    r_u = r_l + 1
    w_l = (r_u - r) * img
    w_u = (r - r_l) * img

    # Do the azimuthal integration as a histogram operation
    r_all = np.concatenate((r_l.flatten(), r_u.flatten())).astype(int)
    if num_dims_img > num_dims_int:
        num_imgs = img.shape[0]
        az_img = []
        for ii in range(num_imgs):
            w_all = np.concatenate((w_l[ii, ...].flatten(), w_u[ii, ...].flatten()))
            az_img.append(np.bincount(r_all, weights=w_all))
        az_img = np.array(az_img)
        return np.reshape(az_img, (*img_old_shape, az_img.shape[-1]))  # type: ignore
    else:
        w_all = np.concatenate((w_l.flatten(), w_u.flatten()))
        return np.bincount(r_all, weights=w_all)


def lines_intersection(
    line_1: NDArray,
    line_2: Union[float, NDArray],
    position: str = "first",
    x_lims: Optional[Tuple[Optional[float], Optional[float]]] = None,
) -> Optional[Tuple[float, float]]:
    """
    Compute the intersection point between two lines.
    Parameters
    ----------
    line_1 : NDArray
        The first line.
    line_2 : float | NDArray
        The second line. It can be a scalar representing a horizontal line.
    position : str, optional
        The position of the point to select. Either "first" or "last".
        The default is "first".
    Raises
    ------
    ValueError
        If position is neither "first" nor "last".
    Returns
    -------
    Tuple[float, float] | None
        It returns either the requested crossing point, or None in case the
        point was not found.
    """
    line_1 = np.array(np.squeeze(line_1), ndmin=1)
    line_2 = np.array(np.squeeze(line_2), ndmin=1)
    # Find the transition points, by first finding where line_2 is above line_1
    crossing_points = np.where(line_2 > line_1, 0, 1)
    crossing_points = np.abs(np.diff(crossing_points))

    if x_lims is not None:
        if x_lims[0] is None:
            if x_lims[1] is None:
                raise ValueError("When passing `x_lims`, at least one of the values should not be None.")
            else:
                bias = 0
                crossing_points = crossing_points[: x_lims[1]]
        else:
            bias = x_lims[0]
            if x_lims[1] is None:
                crossing_points = crossing_points[x_lims[0] :]
            else:
                crossing_points = crossing_points[x_lims[0] : x_lims[1]]
    else:
        bias = 0

    crossing_points = np.where(crossing_points)[0]

    if crossing_points.size == 0:
        print("No crossing found!")
        return None

    if position.lower() == "first":
        point_l = crossing_points[0] + bias
    elif position.lower() == "last":
        point_l = crossing_points[-1] + bias
    else:
        raise ValueError(f"Crossing position: {position} unknown. Please choose either 'first' or 'last'.")

    x1 = 0.0
    x2 = 1.0
    y1 = line_1[point_l]
    y2 = line_1[point_l + 1]
    x3 = 0.0
    x4 = 1.0
    if line_2.size == 1:
        y3 = line_2
        y4 = line_2
    else:
        y3 = line_2[point_l]
        y4 = line_2[point_l + 1]

    # From wikipedia: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line
    p_den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    p_x_num = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
    p_y_num = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)

    p_x = p_x_num / p_den + point_l
    p_y = p_y_num / p_den

    return float(p_x), float(p_y)
    
def frc(
    img1: NDArray,
    img2: NDArray,
    snrt: float = 0.2071,
    axes: Optional[Sequence[int]] = None,
    smooth: Optional[int] = 5,
    taper_ratio: Optional[float] = 0.05,
    supersampling: int = 1,
) -> Tuple[NDArray, NDArray]:
    """
    Compute the FRC/FSC (Fourier ring/shell correlation) between two images / volumes.
    Please refer to the following article for more information:
        M. van Heel and M. Schatz, “Fourier shell correlation threshold criteria,”
        J. Struct. Biol., vol. 151, no. 3, pp. 250–262, Sep. 2005.
    Parameters
    ----------
    img1 : NDArray
        First image / volume.
    img2 : NDArray
        Second image / volume.
    snrt : float, optional
        SNR to be used for generating the threshold curve for resolution definition.
        The SNR value of 0.4142 corresponds to the half-bit curve for a full dataset.
        When splitting datasets in two sub-datasets, that value needs to be halved.
        The default is 0.2071, which corresponds to the half-bit threashold for half dataset.
    axes : Sequence[int], optional
        The axes over which we want to compute the FRC/FSC.
        If None, all axes will be used The default is None.
    smooth : Optional[int], optional
        Size of the Hann smoothing window. The default is 5.
    taper_ratio : Optional[float], optional
        Ratio of the edge pixels to be tapered off. This is necessary when working
        with truncated volumes / local tomography, to avoid truncation artifacts.
        The default is 0.05.
    supersampling : int, optional
        Supersampling factor of the images.
        Larger values increase the high-frequency range of the FRC/FSC function.
        The default is 1, which corresponds to the Nyquist frequency.
    Raises
    ------
    ValueError
        Error returned when not passing images of the same shape.
    Returns
    -------
    NDArray
        The computed FRC/FSC.
    NDArray
        The threshold curve corresponding to the given threshod SNR.
    """
    img1_shape = np.array(img1.shape)

    if axes is None:
        axes = list(np.arange(-len(img1_shape), 0))

    if img2 is None:
        if np.any(img1_shape[axes] % 2 == 1):
            raise ValueError(f"Image shape {img1_shape} along the chosen axes {axes} needs to be even.")
        raise NotImplementedError("Self FRC not implemented, yet.")
    else:
        img2_shape = np.array(img2.shape)
        if len(img1_shape) != len(img2_shape) or np.any(img1_shape != img2_shape):
            raise ValueError(
                f"Image #1 size {img1_shape} and image #2 size {img2_shape} are different, while they should be equal."
            )

    if supersampling > 1:
        base_grid = [np.linspace(-(d - 1) / 2, (d - 1) / 2, d) for d in img1_shape]

        interp_grid = [np.linspace(-(d - 1) / 2, (d - 1) / 2, d) for d in img1_shape]
        for a in axes:
            d = img1_shape[a] * 2
            interp_grid[a] = np.linspace(-(d - 1) / 4, (d - 1) / 4, d)
        interp_grid = np.meshgrid(*interp_grid, indexing="ij")
        interp_grid = np.transpose(interp_grid, [*range(1, len(img1_shape) + 1), 0])

        img1 = sp.interpolate.interpn(base_grid, img1, interp_grid, bounds_error=False, fill_value=None)
        img2 = sp.interpolate.interpn(base_grid, img2, interp_grid, bounds_error=False, fill_value=None)

        img1_shape = np.array(img1.shape)

    axes_shape = img1_shape[list(axes)]
    cut_off = np.min(axes_shape) // 2

    if taper_ratio is not None:
        taper_size = float(taper_ratio * np.mean(axes_shape))
        vol_mask = circular_mask(img1_shape, coords_ball=axes, radius_offset=-taper_size, taper_func="cos")
        img1 = img1 * vol_mask
        img2 = img2 * vol_mask

    img1_f = np.fft.fftn(img1, axes=axes)
    img2_f = np.fft.fftn(img2, axes=axes)

    fc = img1_f * np.conj(img2_f)
    f1 = np.abs(img1_f) ** 2
    f2 = np.abs(img2_f) ** 2

    fc_r_int = azimuthal_integration(fc.real, axes=axes, domain="fourier")
    fc_i_int = azimuthal_integration(fc.imag, axes=axes, domain="fourier")
    fc_int = np.sqrt((fc_r_int**2) + (fc_i_int**2))
    f1_int = azimuthal_integration(f1, axes=axes, domain="fourier")
    f2_int = azimuthal_integration(f2, axes=axes, domain="fourier")

    f1s_f2s = f1_int * f2_int
    f1s_f2s = f1s_f2s + (f1s_f2s == 0)
    f1s_f2s = np.sqrt(f1s_f2s)

    frc = fc_int / f1s_f2s

    rings_size = azimuthal_integration(np.ones_like(img1), axes=axes, domain="fourier")
    # Alternatively:
    # # The number of pixels in a ring is given by the surface.
    # # We compute the n-dimensional hyper-sphere surface, where n is given by the number of axes.
    # n = len(axes)
    # num_surf = 2 * np.pi ** (n / 2)
    # den_surf = sp.special.gamma(n / 2)
    # rings_size = np.concatenate(((1.0, ), num_surf / den_surf * np.arange(1, len(frc)) ** (n - 1)))
    #print(np.sqrt(rings_size))
    Tnum = snrt + (2 * np.sqrt(snrt) + 1) / np.sqrt(rings_size)
    Tden = snrt + 1 + 2 * np.sqrt(snrt) / np.sqrt(rings_size)
    Thb = Tnum / Tden

    if smooth is not None and smooth > 1:
        win = sp.signal.windows.hann(smooth)
        win /= np.sum(win)
        win = win.reshape([*[1] * (frc.ndim - 1), -1])
        frc = sp.ndimage.convolve(frc, win, mode="nearest")

    return frc[..., :cut_off], Thb[..., :cut_off]


def plot_frcs(
    volume_pairs: Sequence[Tuple[NDArray, NDArray]],
    labels: Sequence[str],
    title: Optional[str] = None,
    smooth: Optional[int] = 5,
    snrt: float = 0.2071,
    axes: Optional[Sequence[int]] = None,
    taper_ratio = 0.05,
    verbose: bool = False,
) -> Tuple[Figure, Axes]:
    """Compute and plot the FSCs / FRCs of some volumes.
    Parameters
    ----------
    volume_pairs : Sequence[Tuple[NDArray, NDArray]]
        A list of pairs of volumes to compute the FRCs on.
    labels : Sequence[str]
        The labels associated with each pair.
    title : Optional[str], optional
        The axes title, by default None.
    smooth : Optional[int], optional
        The size of the smoothing window for the computed curves, by default 5.
    snrt : float, optional
        The SNR of the T curve, by default 0.2071 - as per half-dataset SNR.
    axes : Sequence[int] | None, optional
        The axes along which we want to compute the FRC. The unused axes will be
        averaged. The default is None.
    verbose : bool, optional
        Whether to display verbose output, by default False.
    """
    frcs = [np.array([])] * len(volume_pairs)
    xps: List[Optional[Tuple[float, float]]] = [(0.0, 0.0)] * len(volume_pairs)

    for ii, pair in enumerate(tqdm(volume_pairs, desc="Computing FRCs", disable=not verbose)):
        frcs[ii], T = frc(pair[0], pair[1], snrt=snrt, smooth=smooth, axes=axes, taper_ratio=taper_ratio)
        if T.ndim > 1:
            reduce_axes = tuple(np.arange(T.ndim - 1))
            frcs[ii] = frcs[ii].mean(axis=reduce_axes)
            T = T.mean(axis=reduce_axes)
        xps[ii] = lines_intersection(frcs[ii], T, x_lims=(1, None))

    nyquist = len(frcs[0])
    xx = np.linspace(0, 1, nyquist)

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    for f, l in zip(frcs, labels):
        ax.plot(xx, np.squeeze(f), label=l)
    ax.plot(xx, np.squeeze(T), label="T 1/2 bit", linestyle="dashed")
    for ii, p in enumerate(xps):
        if p is not None:
            res = p[0] / (nyquist - 1)
            ax.stem(res, p[1], label=f"Resolution ({labels[ii]}): {res:.3}", linefmt=f"C{ii}-.", markerfmt=f"C{ii}o")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.legend()
    ax.grid()
    ax.set_ylabel("Magnitude")
    ax.set_xlabel("Spatial frequency / Nyquist")
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()

    #plt.show(block=False)

    return fig, ax, res
