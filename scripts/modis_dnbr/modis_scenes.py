r"""The MOD09A1 scene primitives and the selection rules, in one place.

Shared by phase 01, which chooses the pre/post pair, and by phase 08, which
must mask that pair exactly the way the selection filter did -- otherwise a
pixel phase 08 discards would be one the 1% contamination budget never
charged for. Nothing here reads or writes files.

Paths, the grid and the scar rule are in paths.py; the quality bits are in
quality_bits.py.
"""
from __future__ import annotations

# Every function here takes and returns an ee.Image and uses only its
# methods, so the module itself does not need the ee namespace.

# ------------------- the search window and the pair rules -------------------
DATA_RANGE = 400            # DataRange: days searched either side of ignition
MIN_INTERVAL_DAYS = 10      # IntervalloTemporaleMinimoNDBR
MAX_CLOUD_ROI = 1           # at most 1% of the scar pixels contaminated

# How a composite starting EXACTLY on a +/-MIN_INTERVAL_DAYS boundary is
# treated. The perimeter shapefile stores ignition at midnight, so without an
# offset such a composite sits precisely on the boundary and Earth Engine's
# half-open date filter would admit it on the pre side and exclude it on the
# post side. Advancing the reference by one hour makes the boundary unambiguous
# and excludes the composite on both sides, so the pre and post rules are
# symmetric. It affects only composites landing on the boundary to the hour;
# every other selection is identical at 0 or 1. The delivered archive was
# produced with 1.
EVENT_EPSILON_HOURS = 1


def get_qa_bits(image, start, end, new_name):
    """The value of StateQA bits [start, end], shifted down to zero."""
    pattern = 0
    for i in range(start, end + 1):
        pattern += 2 ** i
    return (image.select([0], [new_name])
                 .bitwiseAnd(pattern)
                 .rightShift(start))


def filter_bad_obs_state_qa(image):
    """Reject cloud, cloud shadow, cirrus and snow, matching Landsat.

    The Landsat product rejects QA_PIXEL bits 1-5 and then requires at most 1%
    of the scar to be unrecoverable. These are the MOD09A1 StateQA equivalents
    of those bits, so the 1% threshold is applied to the same set of
    conditions in both products:

        Landsat bit 3 cloud         -> bits 0-1 cloud state, 0 = clear
        Landsat bit 4 cloud shadow  -> bit 2
        Landsat bit 2 cirrus        -> bits 8-9, 0 = none
        Landsat bit 5 snow          -> bit 12 MOD35 snow/ice, bit 15 internal
        Landsat bit 1 dilated cloud -> bit 13 adjacent to cloud

    Both snow bits are tested because they disagree and either one indicates
    snow. QA_RADSAT saturation, which Landsat also excludes, has no MOD09A1
    equivalent and is therefore not represented here.

    Requiring the cloud state to be exactly 0 rejects "mixed" (2) and "not
    set" (3) along with "cloudy" (1): only an explicitly clear pixel is kept.
    """
    qa = image.select('StateQA')
    keep = get_qa_bits(qa, 0, 1, 'cloud_state').eq(0)
    keep = keep.And(get_qa_bits(qa, 2, 2, 'cloud_shadow').eq(0))
    keep = keep.And(get_qa_bits(qa, 8, 9, 'cirrus').eq(0))
    keep = keep.And(get_qa_bits(qa, 12, 12, 'snow_mod35').eq(0))
    keep = keep.And(get_qa_bits(qa, 15, 15, 'snow_internal').eq(0))
    keep = keep.And(get_qa_bits(qa, 13, 13, 'adjacent_cloud').eq(0))
    return image.updateMask(keep)


def add_nbr(image):
    """Add an NBR band: (b02 - b07) / (b02 + b07), near infrared vs SWIR2."""
    nbr = image.normalizedDifference(['sur_refl_b02', 'sur_refl_b07'])
    return image.addBands(nbr.rename('NBR'))


def add_timestamp(image):
    """Add system:time_start as a band, so the pair can be sorted on date."""
    return image.addBands(image.metadata('system:time_start'))
