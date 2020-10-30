import numpy as np
from numba import guvectorize, vectorize, int64, boolean, njit


@vectorize([boolean(int64, int64, int64)], nopython=True)
def dayofyear_range(season_start, season_end, day):
    result = False
    if season_start == season_end:
        result = False
    elif season_start < season_end:
        if day >= season_start and day <= season_end:
            result = True
        else:
            result = False
    elif day >= season_start or day <= season_end:
        result = True
    else:
        result = False
    return result


@guvectorize(
    [(int64[:], int64[:], int64[:], boolean[:, :])], "(n),(n),(m)->(n,m)", nopython=True
)
def dayofyear_range_vec(season_start, season_end, day, result):
    for row in range(0, season_start.shape[0]):
        result[row, :] = dayofyear_range(season_start[row], season_end[row], day)


def dayofyear_checker(season_start, season_end, day):
    """Check whether vector of days is in a vector of ranges.
    """
    result = np.full((season_start.shape[0], day.shape[0]), False)
    dayofyear_range_vec(season_start, season_end, day, result)
    return result

if __name__ == '__main__':
    print("Testing dayofyear module")
    day = np.array([100])
    start = np.array([90, 110, 120, 90, 110])
    end = np.array([110, 120, 110, 80, 90])
    correct_result = np.array([True, False, True, True, False])
    assert((dayofyear_checker(start, end, day).reshape(-1) == correct_result.reshape(-1)).all())
