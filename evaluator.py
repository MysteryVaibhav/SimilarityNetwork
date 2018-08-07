from util import *
from tqdm import tqdm


class Evaluator:
    def __init__(self, params, data_loader):
        self.params = params
        self.data_loader = data_loader
        self.proposed_bbs = json.load(open(self.params.proposed_bbox, "r", encoding="utf-8"))

    def recall_phrase_localization(self, model, is_test=False):
        model.eval()
        data_loader = self.data_loader.phrases_data_loader
        r_1 = 0
        r_5 = 0
        r_10 = 0
        r_max = 0
        count = 0
        for phrase, mask, image, true_bb, img_idxs, _ in tqdm(data_loader):
            if phrase is None:
                continue
            img_idxs = [img_idx.split("#")[0] for img_idx in img_idxs]
            for i, each in enumerate(phrase):
                p = each.unsqueeze(0).expand(self.params.regions_in_image, -1).contiguous()
                m = mask[i].unsqueeze(0).expand(self.params.regions_in_image, -1).contiguous()
                img = image[i]
                s = model(to_variable(p), to_variable(m), to_variable(img), False, True)
                similarity = s.data.cpu().numpy().squeeze(1)
                top_10_img_idx = (-similarity).argsort()[:10]
                if self.is_iou_gt_half(img_idxs[i], true_bb[i], [top_10_img_idx[0]]):
                    r_1 += 1
                    r_5 += 1
                    r_10 += 1
                elif self.is_iou_gt_half(img_idxs[i], true_bb[i], top_10_img_idx[1:5]):
                    r_5 += 1
                    r_10 += 1
                elif self.is_iou_gt_half(img_idxs[i], true_bb[i], top_10_img_idx[6:10]):
                    r_10 += 1
                if self.is_iou_gt_half(img_idxs[i], true_bb[i], similarity.argsort()):
                    r_max += 1
                count += 1
        print("Maximum possible recall, R@{} : {}".format(self.params.regions_in_image, r_max / count))
        return r_1 / count, r_5 / count, r_10 / count

    def is_iou_gt_half(self, img_id, truth_bb, regions):
        proposed_bbs = self.proposed_bbs[img_id]['bboxes']
        for region in regions:
            proposal = proposed_bbs[region]
            for bb in truth_bb:
                if self.iou(bb, proposal) >= 0.5:
                    return True
        return False

    @staticmethod
    def iou(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou