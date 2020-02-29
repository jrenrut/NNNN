def get_IoU(bbox_0, bbox_1):

    xmin_0, ymin_0, xmax_0, ymax_0 = bbox_0
    xmin_1, ymin_1, xmax_1, ymax_1 = bbox_1

    area_0 = float((xmax_0 - xmin_0) * (ymax_0 - ymin_0))
    area_1 = float((xmax_1 - xmin_1) * (ymax_1 - ymin_1))

    xmin = min(xmin_0, xmin_1)
    ymin = min(ymin_0, ymin_1)
    xmax = max(xmax_0, xmax_1)
    ymax = max(ymax_0, ymax_1)

    if xmax < xmin or ymax < ymin:
        return 0.

    area_intersect = float((xmax - xmin) * (ymax - ymin))
    IoU = area_intersect / (area_0 + area_1 - area_intersect)

    return IoU
