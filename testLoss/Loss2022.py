import torch

MAX_FLOW = 400

def sequence_loss2022(flow_preds, flow_gt, valid):
    """ Loss function defined over sequence of flow predictions """

    gamma = 0.85
    max_flow = 400
    # 1
    n_predictions = len(flow_preds)

    flow_loss = 0.0
    flow_gt_thresholds = [5, 10, 20]

    # exlude invalid pixels and extremely large diplacements
    # 1 256 1024
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    # 1 256 1024
    valid = (valid >= 0.5) & (mag < max_flow)
    # 1 1 256 1024
    # print("valid.shape", valid[:, None].shape)
    for i in range(n_predictions):
        # 1.0
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        # 返回乘积后的平均值
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    # epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    # epe = epe.view(-1)[valid.view(-1)]
    #
    # metrics = {
    #     'epe': epe.mean().item(),
    #     '1px': (epe < 1).float().mean().item(),
    #     '3px': (epe < 3).float().mean().item(),
    #     '5px': (epe < 5).float().mean().item(),
    # }
    #
    # flow_gt_length = torch.sum(flow_gt**2, dim=1).sqrt()
    # flow_gt_length = flow_gt_length.view(-1)[valid.view(-1)]
    # for t in flow_gt_thresholds:
    #     e = epe[flow_gt_length < t]
    #     metrics.update({
    #             f"{t}-th-5px": (e < 5).float().mean().item()
    #     })


    return flow_loss

