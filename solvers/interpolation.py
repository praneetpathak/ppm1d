"""
PPM Interpolation Kernels (intrf0)

Pure NumPy implementation of PPM interpolation.
Based on PPMstar Fortran implementation.
"""

import numpy as np


def intrf0_pass1_slopes(uy, duysppmzr, duymnotzr):
    """
    Compute central differences and Van Leer monotonized slopes.

    Args:
        uy: Input field array (n_total,)
        duysppmzr: Output central differences (n_total,) - preallocated
        duymnotzr: Output monotonized slopes (n_total,) - preallocated
    """
    n_total = len(uy)
    for i in range(1, n_total - 1):
        # Forward and backward differences
        duyr_i = uy[i+1] - uy[i]
        duyl_i = uy[i] - uy[i-1]

        # Central difference (PPM slope)
        duysppmzr[i] = 0.5 * (uy[i+1] - uy[i-1])

        # Van Leer monotonization of the central difference
        s_ = 1.0 if duysppmzr[i] >= 0 else -1.0

        # Take minimum of |forward diff|, |backward diff|, |central diff|
        thngy1_ = s_ * duyl_i
        thngy2_ = s_ * duyr_i
        if thngy2_ < thngy1_:
            thngy1_ = thngy2_
        thngy1_ = 2.0 * thngy1_
        thngy2_ = s_ * duysppmzr[i]
        if thngy2_ < thngy1_:
            thngy1_ = thngy2_
        if thngy1_ < 0:
            thngy1_ = 0
        duymnotzr[i] = s_ * thngy1_


def intrf0_pass1_interfaces(uy, duysppmzr, duymnotzr, uyrsmth, uyrunsm):
    """
    Compute smooth and unsmooth interface values.
    Must be called AFTER intrf0_pass1_slopes completes.

    Args:
        uy: Input field array
        duysppmzr: Central differences from pass 1
        duymnotzr: Monotonized slopes from pass 1
        uyrsmth: Output smooth interface values
        uyrunsm: Output unsmooth interface values
    """
    n_total = len(uy)
    for i in range(1, n_total - 1):
        duyr_i = uy[i+1] - uy[i]
        thyng = uy[i] + 0.5 * duyr_i

        if i >= 2:
            duysppm_prev = duysppmzr[i-1]
            duymnot_prev = duymnotzr[i-1]
        else:
            duysppm_prev = duysppmzr[i]
            duymnot_prev = duymnotzr[i]

        uyrsmth[i] = thyng - (1.0/6.0) * (duysppmzr[i] - duysppm_prev)
        uyrunsm[i] = thyng - (1.0/6.0) * (duymnotzr[i] - duymnot_prev)


def intrf0_pass2(uy, smaldu, unsmuy, uyrsmth, uyrunsm, uyl, uyr, duy, uy6,
                 unsmoothness_scale=10.0, unsmoothness_offset=0.1):
    """
    Apply monotonicity constraints and compute final interface values.

    Args:
        uy: Input field array
        smaldu: Small delta values for division safety
        unsmuy: Unsmoothness array (modified in-place)
        uyrsmth: Smooth interface values from pass 1
        uyrunsm: Unsmooth interface values from pass 1
        uyl: Output left interface values
        uyr: Output right interface values
        duy: Output delta (uyr - uyl)
        uy6: Output parabola curvature
        unsmoothness_scale: Scale factor for unsmoothness detection
        unsmoothness_offset: Offset for unsmoothness threshold
    """
    n_total = len(uy)
    for i in range(2, n_total - 2):
        # Get interface values from Pass 1
        uylsmth_i = uyrsmth[i-1]
        uylunsm_i = uyrunsm[i-1]
        uyrsmth_i = uyrsmth[i]
        uyrunsm_i = uyrunsm[i]

        # 5-point stencil unsmoothness computation
        duyl_1 = uy[i] - uy[i-1]
        duyr_1 = uy[i+1] - uy[i]

        if i >= 3:
            duyr_0 = uy[i-1] - uy[i-2]
        else:
            duyr_0 = duyl_1

        if i < n_total - 3:
            duyl_2 = uy[i+2] - uy[i+1]
        else:
            duyl_2 = duyr_1

        adiff = uy[i+1] - uy[i-1]
        azrdif = 3.0 * duyl_1 - duyr_0 - adiff
        azldif = duyl_2 - 3.0 * duyr_1 + adiff

        ferror = 0.5 * (abs(azldif) + abs(azrdif)) / (abs(duyl_1) + abs(duyr_1) + smaldu[i])
        unsmooth_ = unsmoothness_scale * (ferror - unsmoothness_offset)
        unsmooth_ = max(0.0, min(1.0, unsmooth_))

        # Update unsmoothness in place
        if unsmooth_ > unsmuy[i]:
            unsmuy[i] = unsmooth_

        # Use the updated unsmoothness
        local_unsm = unsmuy[i]

        # Start with unsmoothed (monotonized) interface values
        uyl_i = uylunsm_i
        uyr_i = uyrunsm_i

        # Apply monotonicity constraints
        almon_ = 3.0 * uy[i] - 2.0 * uyr_i
        armon_ = 3.0 * uy[i] - 2.0 * uyl_i

        # Check if cell i is a local extremum
        if (uy[i] - uyl_i) * (uy[i] - uyr_i) >= 0:
            uyl_i = uy[i]
            uyr_i = uy[i]
            almon_ = uy[i]
            armon_ = uy[i]

        # Additional monotonicity constraints
        if (uyr_i - uyl_i) * (almon_ - uyl_i) > 0:
            uyl_i = almon_
        if (uyr_i - uyl_i) * (armon_ - uyr_i) < 0:
            uyr_i = armon_

        # Blend smooth and unsmooth versions
        uyl[i] = uylsmth_i - local_unsm * (uylsmth_i - uyl_i)
        uyr[i] = uyrsmth_i - local_unsm * (uyrsmth_i - uyr_i)
        duy[i] = uyr[i] - uyl[i]
        uy6[i] = 6.0 * (uy[i] - 0.5 * (uyl[i] + uyr[i]))
