import numpy as np

"""
This file contains the code to perform the linking of particle between
image frames. Nothing should be altered in this file if you are simply
trying to run the code.
"""


def finding_indices(x_ind, xPred_ind, y_ind, yPred_ind):
    """
    Finds the x and y locations that have the same indices for each array
    Inputs: x_ind - indices of the original x locations
            xPred_ind - indices of the predicted x locations
            y_ind - indices of the original y locations
            yPred_ind - indices of the predicted y locations
    Outputs: ind - indices of the same original x and y locations
             ind_pred - indices of the same predicted x and y locations
    """
    a = np.zeros((len(x_ind), 2), dtype=np.int)
    a[:, 0] = x_ind
    a[:, 1] = xPred_ind

    b = np.zeros((len(y_ind), 2), dtype=np.int)
    b[:, 0] = y_ind
    b[:, 1] = yPred_ind

    nrows, ncols = a.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
             'formats': ncols * [a.dtype]}

    # The following line returns the (x,y) values that are the same in both
    C = np.intersect1d(a.view(dtype), b.view(dtype))

    D = np.in1d(a.view(dtype), b.view(dtype))

    a = a[D]

    ind = a[:, 0]
    ind_pred = a[:, 1]

    return ind, ind_pred


def previous_tracks(data, im, ii, box_size):
    """
    Runs the particle tracking code for a path that has already been started
    Inputs: data - the data array containing information about particles 
                   (size, location, etc) and previous tracking results
            im - the current image number
            ii - the current particle in the image (im)
            box_size - size of the search box to use
    Outputs: data - the data array containing information about particles and
                    previous tracking results, now updated for the current
                    particle
    """
    im0 = np.where(data.Slice == im - 1)[0]
    im1 = np.where(data.Slice == im)[0]
    im2 = np.where(data.Slice == im + 1)[0]
    im3 = np.where(data.Slice == im + 2)[0]

    x0 = data.x[im0[data.Count[im1[ii]] == data.Count[im0]]]
    y0 = data.y[im0[data.Count[im1[ii]] == data.Count[im0]]]

    xPred2 = 2 * data.x[im1[ii]] - x0
    yPred2 = 2 * data.y[im1[ii]] - y0

    x2_ind = np.where((data.x[im2] >= xPred2 - box_size) &
                      (data.x[im2] <= xPred2 + box_size))
    y2_ind = np.where((data.y[im2] >= yPred2 - box_size) &
                      (data.y[im2] <= yPred2 + box_size))

    ind2 = np.intersect1d(x2_ind[0], y2_ind[0])

    if len(ind2) == 0:
        temp_loc = np.where(data.Count[im1[ii]] == data.CountTemp[im2])[0]
        if len(temp_loc) == 1:
            data.Count[im2[temp_loc]] = data.CountTemp[im2[temp_loc]]
        return data

    xPred3 = 2.5 * data.x[im2[ind2]] - 2 * data.x[im1[ii]] + 0.5 * x0
    yPred3 = 2.5 * data.y[im2[ind2]] - 2 * data.y[im1[ii]] + 0.5 * y0

    xPred3_gr, x3_gr = np.meshgrid(xPred3, data.x[im3])
    yPred3_gr, y3_gr = np.meshgrid(yPred3, data.y[im3])

    x3_ind, xPred3_ind = np.where((x3_gr >= xPred3_gr - box_size) &
                                  (x3_gr <= xPred3_gr + box_size))
    y3_ind, yPred3_ind = np.where((y3_gr >= yPred3_gr - box_size) &
                                  (y3_gr <= yPred3_gr + box_size))

    ind3, ind3_pred = finding_indices(x3_ind, xPred3_ind, y3_ind, yPred3_ind)

    if len(ind3) == 0:
        temp_loc = np.where(data.Count[im1[ii]] == data.CountTemp[im2])[0]
        if len(temp_loc) == 1:
            data.Count[im2[temp_loc]] = data.CountTemp[im2[temp_loc]]
        return data

    cost = np.sqrt((xPred3_gr[ind3, ind3_pred] - x3_gr[ind3, ind3_pred])**2 +
                   (yPred3_gr[ind3, ind3_pred] - y3_gr[ind3, ind3_pred])**2)

    min_cost = np.where(cost == cost.min())[0]

    if len(min_cost) > 1:
        return data

    if data.Count[im2[ind2[ind3_pred[min_cost]]]] == 0:
        count = data.Count[im1[ii]]
        data.Count[im2[ind2[ind3_pred[min_cost]]]] = count
        data.Cost[im2[ind2[ind3_pred[min_cost]]]] = cost[min_cost]
        data.CountTemp[im3[ind3[min_cost]]] = count
    else:
        ind_bad = np.where(data.Count[im2[ind2[ind3_pred[min_cost]]]] ==
                           data.Count[im2])[0]
        if cost.min() < data.Cost[im2[ind_bad]]:
            count = data.Count[im1[ii]]
            data.Count[im2[ind2[ind3_pred[min_cost]]]] = count
            data.Cost[im2[ind2[ind3_pred[min_cost]]]] = cost[min_cost]
            data.CountTemp[im3[ind3[min_cost]]] = count

    return data


def no_previous_tracks(data, im, ii, box_size, box_size_initial_x, box_size_initial_y):
    """
    Runs the particle tracking code for a path that has not already been started
    Inputs: data - the data array containing information about particles
                   (size, location, etc) and previous tracking results
            im - the current image number
            ii - the current particle in the image (im)
            box_size - size of the search box to use
    Outputs: data - the data array containing information about particles and
                    previous tracking results, now updated for the current
                    particle
    """
    im0 = np.where(data.Slice == im)[0]
    im1 = np.where(data.Slice == im + 1)[0]
    im2 = np.where(data.Slice == im + 2)[0]
    im3 = np.where(data.Slice == im + 3)[0]

    x1_ind = np.where((data.x[im1] >= data.x[im0[ii]] - box_size_initial_x) &
                      (data.x[im1] <= data.x[im0[ii]] + box_size_initial_x))[0]
    y1_ind = np.where((data.y[im1] >= data.y[im0[ii]] - box_size_initial_y) &
                      (data.y[im1] <= data.y[im0[ii]] + box_size_initial_y))[0]

    ind1 = np.intersect1d(x1_ind, y1_ind)

    if len(ind1) == 0:
        return data

    xPred2 = 2 * data.x[im1[ind1]] - data.x[im0[ii]]
    yPred2 = 2 * data.y[im1[ind1]] - data.y[im0[ii]]

    xPred2_gr, x2_gr = np.meshgrid(xPred2, data.x[im2])
    yPred2_gr, y2_gr = np.meshgrid(yPred2, data.y[im2])

    x2_ind, xPred2_ind = np.where((x2_gr >= xPred2_gr - box_size) &
                                  (x2_gr <= xPred2_gr + box_size))
    y2_ind, yPred2_ind = np.where((y2_gr >= yPred2_gr - box_size) &
                                  (y2_gr <= yPred2_gr + box_size))

    ind2, ind2_pred = finding_indices(x2_ind, xPred2_ind, y2_ind, yPred2_ind)

    if len(ind2) == 0:
        return data

    xPred3 = 2.5 * data.x[im2[ind2]] - 2 * data.x[im1[ind1[ind2_pred]]] + 0.5 * data.x[im0[ii]]
    yPred3 = 2.5 * data.y[im2[ind2]] - 2 * data.y[im1[ind1[ind2_pred]]] + 0.5 * data.y[im0[ii]]

    xPred3_gr, x3_gr = np.meshgrid(xPred3, data.x[im3])
    yPred3_gr, y3_gr = np.meshgrid(yPred3, data.y[im3])

    x3_ind, xPred3_ind = np.where((x3_gr >= xPred3_gr - box_size) &
                                  (x3_gr <= xPred3_gr + box_size))
    y3_ind, yPred3_ind = np.where((y3_gr >= yPred3_gr - box_size) &
                                  (y3_gr <= yPred3_gr + box_size))

    ind3, ind3_pred = finding_indices(x3_ind, xPred3_ind, y3_ind, yPred3_ind)

    if len(ind3) == 0:
        return data

    cost = np.sqrt((xPred3_gr[ind3, ind3_pred] - x3_gr[ind3, ind3_pred])**2 +
                   (yPred3_gr[ind3, ind3_pred] - y3_gr[ind3, ind3_pred])**2)

    min_cost = np.where(cost == cost.min())[0]

    if len(min_cost) > 1:
        return data

    if data.Count[im1[ind1[ind2_pred[ind3_pred[min_cost]]]]] == 0:
        count = data.Count.max() + 1
        data.Count[im0[ii]] = count
        data.Count[im1[ind1[ind2_pred[ind3_pred[min_cost]]]]] = count
        data.Cost[im1[ind1[ind2_pred[ind3_pred[min_cost]]]]] = cost[min_cost]
        data.CountTemp[im2[ind2[ind3_pred[min_cost]]]] = count
    else:
        ind_bad = np.where(data.Count[im1[ind1[ind2_pred[ind3_pred[min_cost]]]]] ==
                           data.Count[im1])[0]
        if cost.min() < data.Cost[im1[ind_bad]]:
            ind_bad_2 = np.where(data.Count[im1[ind_bad]] ==
                                 data.CountTemp[im2])[0]
            count = data.Count.max() + 1
            data.Count[im0[ii]] = count
            data.Count[im1[ind1[ind2_pred[ind3_pred[min_cost]]]]] = count
            data.Cost[im1[ind1[ind2_pred[ind3_pred[min_cost]]]]] = cost[min_cost]
            data.CountTemp[im2[ind2[ind3_pred[min_cost]]]] = count

    return data


def previous_tracks_3d(data, im, ii, box_size):
    """
    Runs the particle tracking code for a path that has already been started
    Inputs: data - the data array containing information about particles
                   (size, location, etc) and previous tracking results
            im - the current image number
            ii - the current particle in the image (im)
            box_size - size of the search box to use
    Outputs: data - the data array containing information about particles and
                    previous tracking results, now updated for the current
                    particle
    """
    im0 = np.where(data.Slice == im - 1)[0]
    im1 = np.where(data.Slice == im)[0]
    im2 = np.where(data.Slice == im + 1)[0]
    im3 = np.where(data.Slice == im + 2)[0]

    x0 = data.x[im0[data.Count[im1[ii]] == data.Count[im0]]]
    y0 = data.y[im0[data.Count[im1[ii]] == data.Count[im0]]]
    z0 = data.z[im0[data.Count[im1[ii]] == data.Count[im0]]]

    xPred2 = 2 * data.x[im1[ii]] - x0
    yPred2 = 2 * data.y[im1[ii]] - y0
    zPred2 = 2 * data.z[im1[ii]] - z0

    x2_ind = np.where((data.x[im2] >= xPred2 - box_size) &
                      (data.x[im2] <= xPred2 + box_size))
    y2_ind = np.where((data.y[im2] >= yPred2 - box_size) &
                      (data.y[im2] <= yPred2 + box_size))
    z2_ind = np.where((data.z[im2] >= zPred2 - box_size) &
                      (data.z[im2] <= zPred2 + box_size))

    ind2_ = np.intersect1d(x2_ind[0], y2_ind[0])
    ind2 = np.intersect1d(ind2_, z2_ind[0])

    if len(ind2) == 0:
        temp_loc = np.where(data.Count[im1[ii]] == data.CountTemp[im2])[0]
        if len(temp_loc) == 1:
            data.Count[im2[temp_loc]] = data.CountTemp[im2[temp_loc]]
            return data

    xPred3 = 2.5 * data.x[im2[ind2]] - 2 * data.x[im1[ii]] + 0.5 * x0
    yPred3 = 2.5 * data.y[im2[ind2]] - 2 * data.y[im1[ii]] + 0.5 * y0
    zPred3 = 2.5 * data.z[im2[ind2]] - 2 * data.z[im1[ii]] + 0.5 * z0

    xPred3_gr, x3_gr = np.meshgrid(xPred3, data.x[im3])
    yPred3_gr, y3_gr = np.meshgrid(yPred3, data.y[im3])
    zPred3_gr, z3_gr = np.meshgrid(zPred3, data.z[im3])

    x3_ind, xPred3_ind = np.where((x3_gr >= xPred3_gr - box_size) &
                                  (x3_gr <= xPred3_gr + box_size))
    y3_ind, yPred3_ind = np.where((y3_gr >= yPred3_gr - box_size) &
                                  (y3_gr <= yPred3_gr + box_size))
    z3_ind, zPred3_ind = np.where((z3_gr >= zPred3_gr - box_size) &
                                  (z3_gr <= zPred3_gr + box_size))

    ind3_, ind3_pred_ = finding_indices(x3_ind, xPred3_ind, y3_ind, yPred3_ind)
    ind3, ind3_pred = finding_indices(ind3_, ind3_pred_, z3_ind, zPred3_ind)

    if len(ind3) == 0:
        temp_loc = np.where(data.Count[im1[ii]] == data.CountTemp[im2])[0]
        if len(temp_loc) == 1:
            data.Count[im2[temp_loc]] = data.CountTemp[im2[temp_loc]]
        return data

    cost = np.sqrt((xPred3_gr[ind3, ind3_pred] - x3_gr[ind3, ind3_pred])**2 +
                   (yPred3_gr[ind3, ind3_pred] - y3_gr[ind3, ind3_pred])**2 +
                   (zPred3_gr[ind3, ind3_pred] - z3_gr[ind3, ind3_pred])**2)

    min_cost = np.where(cost == cost.min())[0]

    if len(min_cost) > 1:
        return data

    if data.Count[im2[ind2[ind3_pred[min_cost]]]] == 0:
        count = data.Count[im1[ii]]
        data.Count[im2[ind2[ind3_pred[min_cost]]]] = count
        data.Cost[im2[ind2[ind3_pred[min_cost]]]] = cost[min_cost]
        data.CountTemp[im3[ind3[min_cost]]] = count
    else:
        ind_bad = np.where(data.Count[im2[ind2[ind3_pred[min_cost]]]] ==
                           data.Count[im2])[0]
        if cost.min() < data.Cost[im2[ind_bad]]:
            ind_bad_2 = np.where(data.CountTemp[im3[ind3[min_cost]]] ==
                                 data.CountTemp[im3])[0]
            count = data.Count[im1[ii]]
            data.Count[im2[ind2[ind3_pred[min_cost]]]] = count
            data.Cost[im2[ind2[ind3_pred[min_cost]]]] = cost[min_cost]
            data.CountTemp[im3[ind3[min_cost]]] = count
            data.Count[im2[ind_bad]] = 0
            data.CountTemp[im3[ind_bad_2]] = 0

    return data


def no_previous_tracks_3d(data, im, ii, box_size, box_size_initial_x, box_size_initial_y, box_size_initial_z):
    """
    Runs the particle tracking code for a path that has not already been started
    Inputs: data - the data array containing information about particles
                   (size, location, etc) and previous tracking results
            im - the current image number
            ii - the current particle in the image (im)
            box_size - size of the search box to use
    Outputs: data - the data array containing information about particles and
                    previous tracking results, now updated for the current
                    particle
    """
    im0 = np.where(data.Slice == im)[0]
    im1 = np.where(data.Slice == im + 1)[0]
    im2 = np.where(data.Slice == im + 2)[0]
    im3 = np.where(data.Slice == im + 3)[0]

    x1_ind = np.where((data.x[im1] >= data.x[im0[ii]] - box_size_initial_x) &
                      (data.x[im1] <= data.x[im0[ii]] + box_size_initial_x))[0]
    y1_ind = np.where((data.y[im1] >= data.y[im0[ii]] - box_size_initial_y) &
                      (data.y[im1] <= data.y[im0[ii]] + box_size_initial_y))[0]
    z1_ind = np.where((data.z[im1] >= data.z[im0[ii]] - box_size_initial_z) &
                      (data.z[im1] <= data.z[im0[ii]] + box_size_initial_z))[0]

    ind1_ = np.intersect1d(x1_ind, y1_ind)
    ind1 = np.intersect1d(ind1_, z1_ind)

    if len(ind1) == 0:
        return data

    xPred2 = 2 * data.x[im1[ind1]] - data.x[im0[ii]]
    yPred2 = 2 * data.y[im1[ind1]] - data.y[im0[ii]]
    zPred2 = 2 * data.z[im1[ind1]] - data.z[im0[ii]]

    xPred2_gr, x2_gr = np.meshgrid(xPred2, data.x[im2])
    yPred2_gr, y2_gr = np.meshgrid(yPred2, data.y[im2])
    zPred2_gr, z2_gr = np.meshgrid(zPred2, data.z[im2])

    x2_ind, xPred2_ind = np.where((x2_gr >= xPred2_gr - box_size) &
                                  (x2_gr <= xPred2_gr + box_size))
    y2_ind, yPred2_ind = np.where((y2_gr >= yPred2_gr - box_size) &
                                  (y2_gr <= yPred2_gr + box_size))
    z2_ind, zPred2_ind = np.where((z2_gr >= zPred2_gr - box_size) &
                                  (z2_gr <= zPred2_gr + box_size))

    ind2_, ind2_pred_ = finding_indices(x2_ind, xPred2_ind, y2_ind, yPred2_ind)
    ind2, ind2_pred = finding_indices(ind2_, ind2_pred_, z2_ind, zPred2_ind)

    if len(ind2) == 0:
        return data

    xPred3 = 2.5 * data.x[im2[ind2]] - 2 * data.x[im1[ind1[ind2_pred]]] + 0.5 * data.x[im0[ii]]
    yPred3 = 2.5 * data.y[im2[ind2]] - 2 * data.y[im1[ind1[ind2_pred]]] + 0.5 * data.y[im0[ii]]
    zPred3 = 2.5 * data.z[im2[ind2]] - 2 * data.z[im1[ind1[ind2_pred]]] + 0.5 * data.z[im0[ii]]

    xPred3_gr, x3_gr = np.meshgrid(xPred3, data.x[im3])
    yPred3_gr, y3_gr = np.meshgrid(yPred3, data.y[im3])
    zPred3_gr, z3_gr = np.meshgrid(zPred3, data.z[im3])

    x3_ind, xPred3_ind = np.where((x3_gr >= xPred3_gr - box_size) &
                                  (x3_gr <= xPred3_gr + box_size))
    y3_ind, yPred3_ind = np.where((y3_gr >= yPred3_gr - box_size) &
                                  (y3_gr <= yPred3_gr + box_size))
    z3_ind, zPred3_ind = np.where((z3_gr >= zPred3_gr - box_size) &
                                  (z3_gr <= zPred3_gr + box_size))

    ind3_, ind3_pred_ = finding_indices(x3_ind, xPred3_ind, y3_ind, yPred3_ind)
    ind3, ind3_pred = finding_indices(ind3_, ind3_pred_, z3_ind, zPred3_ind)

    if len(ind3) == 0:
        return data

    cost = np.sqrt((xPred3_gr[ind3, ind3_pred] - x3_gr[ind3, ind3_pred])**2 +
                   (yPred3_gr[ind3, ind3_pred] - y3_gr[ind3, ind3_pred])**2 +
                   (zPred3_gr[ind3, ind3_pred] - z3_gr[ind3, ind3_pred])**2)

    min_cost = np.where(cost == cost.min())[0]

    if len(min_cost) > 1:
        return data

    if data.Count[im1[ind1[ind2_pred[ind3_pred[min_cost]]]]] == 0:
        count = data.Count.max() + 1
        data.Count[im0[ii]] = count
        data.Count[im1[ind1[ind2_pred[ind3_pred[min_cost]]]]] = count
        data.Cost[im1[ind1[ind2_pred[ind3_pred[min_cost]]]]] = cost[min_cost]
        data.CountTemp[im2[ind2[ind3_pred[min_cost]]]] = count
    else:
        ind_bad = np.where(data.Count[im1[ind1[ind2_pred[ind3_pred[min_cost]]]]] ==
                           data.Count[im1])[0]
        if cost.min() < data.Cost[im1[ind_bad]]:
            ind_bad_2 = np.where(data.CountTemp[im2[ind2[ind3_pred[min_cost]]]] ==
                                 data.CountTemp[im2])[0]
            count = data.Count.max() + 1
            data.Count[im0[ii]] = count
            data.Count[im1[ind1[ind2_pred[ind3_pred[min_cost]]]]] = count
            data.Cost[im1[ind1[ind2_pred[ind3_pred[min_cost]]]]] = cost[min_cost]
            data.CountTemp[im2[ind2[ind3_pred[min_cost]]]] = count

    return data
