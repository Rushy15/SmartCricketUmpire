def check_lbw(ball_position, pad_box, bat_box, stumps_box):
    lbw = False
    in_pad = False

    if pad_box:
        if pad_box[0] <= ball_position[0] <= pad_box[2] and pad_box[1] <= ball_position[1] <= pad_box[3]:
            in_pad = True

    bat_overlap = False
    if bat_box:
        bx1, by1, bx2, by2 = bat_box
        if bx1 <= ball_position[0] <= bx2 and by1 <= ball_position[1] <= by2:
            bat_overlap = True

    stump_check = False
    if stumps_box:
        x1, y1, x2, y2 = stumps_box
        if x1 <= ball_position[0] <= x2 and y1 <= ball_position[1] <= y2:
            stump_check = True

    # Modify logic to allow LBW even if pad_box is not present or in_pad is False
    if stump_check and not bat_overlap:
        lbw = True

    return lbw
