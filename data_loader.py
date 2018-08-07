import torch.utils.data
from util import *


class CustomDataSet(torch.utils.data.TensorDataset):
    def __init__(self, img_one_hot, ids, regions_in_image, visual_feature_dimension, image_features_dir, entity_sent,
                 mapping):
        self.img_one_hot = img_one_hot
        self.ids = ids
        self.num_of_samples = len(ids)
        self.regions_in_image = regions_in_image
        self.visual_feature_dimension = visual_feature_dimension
        self.image_features_dir = image_features_dir
        self.mapping = mapping
        self.entity_sent = entity_sent

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, idx):
        #image = np.random.random((self.regions_in_image, self.visual_feature_dimension))
        image = np.zeros((self.regions_in_image, self.visual_feature_dimension))
        image_ = np.load(self.image_features_dir + "{}.npy".format(self.ids[idx].split("#")[0])).reshape(
                       (-1, self.visual_feature_dimension))
        image[:image_.shape[0],:] = image_
        phrases = self.entity_sent[self.ids[idx]]["entities"]
        return image, self.ids[idx], phrases
    
    def collate(self, batch):
        images = [x[0] for x in batch]
        img_idxs = [x[1] for x in batch]
        phrases = [x[2] for x in batch]

        return self.make_phrase_mini_batch(img_idxs, phrases, images)

    def get_image_feature(self, image, key):
        feat = image[self.mapping[key][0]]
        for i in range(1, len(self.mapping[key])):
            feat += image[self.mapping[key][i]]
        return feat / len(key)

    def make_phrase_mini_batch(self, img_idxs, phrases, images):
        # Construct batches for phrases
        input = []
        img = []
        label = []

        for i, img_idx in enumerate(img_idxs):
            for phrases_in_batch in phrases:
                types = []
                phrase_inputs = []
                phrase_images = []
                for j, phrase in enumerate(phrases_in_batch):
                    key = img_idx.split("#")[0] + ":" + phrase[0]
                    phrase_len = len(phrase)
                    if key in self.mapping:
                        start_idx = phrase[phrase_len - 2]
                        end_idx = phrase[phrase_len - 1]
                        img_feature = self.get_image_feature(images[i], key)
                        phrase_input, _ = get_phrase_encoding(img_idx, start_idx, end_idx)
                        phrase_images.append(img_feature)
                        phrase_inputs.append(phrase_input)
                        types.append(phrase[phrase_len - 3][0])
                if self.different_types(types):
                    for idx in range(0, len(phrase_inputs)):
                        r_n_idx = idx
                        while types[r_n_idx] == types[idx]:
                            r_n_idx = np.random.randint(len(phrase_inputs))
                        input.append(phrase_inputs[idx])
                        input.append(phrase_inputs[idx])
                        img.append(phrase_images[idx])
                        img.append(phrase_images[r_n_idx])
                        label.append(1)
                        label.append(-1)
        return self.batchify(input, img, label)

    @staticmethod
    def different_types(types):
        if len(types) == 0:
            return False
        i = 0
        type = types[i]
        while i < len(types) and types[i] == type:
            i += 1
        return False if i == len(types) else True

    @staticmethod
    def batchify(input, img, label):
        max_len = np.max([x.shape[0] for x in input])
        padded_input = np.zeros((len(input), max_len))
        mask = np.zeros((len(input), max_len))
        i = 0
        for each in input:
            padded_input[i, :len(each)] = each
            mask[i, :len(each)] = 1
            i += 1
        return to_tensor(padded_input).long(), to_tensor(mask), to_tensor(np.array(img)), to_tensor(np.array(label))


class CustomDataSet3(torch.utils.data.TensorDataset):
    def __init__(self, ids, regions_in_image, visual_feature_dimension, image_features_dir, entity_sent, entity_bbox):
        self.ids = ids
        self.num_of_samples = len(ids)
        self.regions_in_image = regions_in_image
        self.visual_feature_dimension = visual_feature_dimension
        self.image_features_dir = image_features_dir
        self.entity_sent = entity_sent
        self.entity_bbox = entity_bbox

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, idx):

        #image = np.random.random((self.regions_in_image, self.visual_feature_dimension))
        image = np.zeros((self.regions_in_image, self.visual_feature_dimension))
        image_ = np.load(self.image_features_dir + "{}.npy".format(self.ids[idx].split("#")[0])).reshape(
                      (-1, self.visual_feature_dimension))
        image[:image_.shape[0],:] = image_

        phrases = self.entity_sent[self.ids[idx]]["entities"]
        bboxes = self.entity_bbox[self.ids[idx].split("#")[0]]['objects']
        return image, self.ids[idx], phrases, bboxes

    def collate(self, batch):
        images = [x[0] for x in batch]
        img_idxs = [x[1] for x in batch]
        phrases = [x[2] for x in batch]
        bbs = [x[3] for x in batch]
        input_phrase = []
        input_image = []
        truth_bb = []
        phrases_text = []
        ids = []
        for i, img_idx in enumerate(img_idxs):
            ground_truth_bb = self.process_truth_bb(bbs[i])
            for phrases_in_batch in phrases:
                for j, phrase in enumerate(phrases_in_batch):
                    phrase_len = len(phrase)
                    if phrase[0] in ground_truth_bb:
                        start_idx = phrase[phrase_len - 2]
                        end_idx = phrase[phrase_len - 1]
                        c, phrase_input = get_phrase_encoding(img_idx, start_idx, end_idx, return_caption=True)
                        bb = ground_truth_bb[phrase[0]]
                        input_phrase.append(phrase_input)
                        input_image.append(images[i])
                        truth_bb.append(bb)
                        ids.append(img_idx)
                        phrases_text.append(c)
        return self.batchify(input_phrase, input_image, truth_bb, ids, phrases_text)

    @staticmethod
    def process_truth_bb(bbs):
        truth = {}
        for each in bbs:
            if 'bbox' in each:
                for chain_id in each['chainIds']:
                    if chain_id not in truth:
                        truth[chain_id] = []
                    truth[chain_id].append(each['bbox'])
        return truth

    @staticmethod
    def batchify(input_phrase, image, bb, img_idxs, phrases_text):
        if len(input_phrase) == 0:
            return None, None, None, None, None, None
        max_len = np.max([x.shape[0] for x in input_phrase])
        padded_input_pos = np.zeros((len(input_phrase), max_len))
        mask_pos = np.zeros((len(input_phrase), max_len))
        i = 0
        for each in input_phrase:
            padded_input_pos[i, :len(each)] = each
            mask_pos[i, :len(each)] = 1
            i += 1
        return to_tensor(padded_input_pos).long(), to_tensor(mask_pos), to_tensor(np.array(image)), bb, img_idxs, phrases_text


class DataLoader:
    def __init__(self, params):
        self.params = params
        self.img_one_hot = run(params.caption_file)
        self.train_ids = get_ids('train', params.split_file)
        self.val_ids = get_ids('val', params.split_file)
        self.plain_val_ids = get_ids('val', params.split_file, strip=True)
        self.test_ids = get_ids('test', params.split_file)
        self.plain_test_ids = get_ids('test', params.split_file, strip=True)
        self.regions_in_image = params.regions_in_image
        self.max_caption_len = params.max_caption_len
        self.entity_sent = json.load(open(params.entity_sent, "r", encoding="utf-8"))
        self.entity_bbox = json.load(open(params.entity_bbox, "r", encoding="utf-8"))
        self.mapping = np.load('mapping.npy').item()
        kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
        #kwargs = {} if torch.cuda.is_available() else {}
        cd1 = CustomDataSet(self.img_one_hot,
                            self.train_ids,
                            params.regions_in_image,
                            params.visual_feature_dimension,
                            params.image_features_dir,
                            self.entity_sent,
                            self.mapping)
        self.training_data_loader = torch.utils.data.DataLoader(cd1,
                                                                batch_size=self.params.batch_size,
                                                                collate_fn=cd1.collate,
                                                                shuffle=True, **kwargs)
        cd2 = CustomDataSet3(self.test_ids,
                            params.regions_in_image,
                            params.visual_feature_dimension,
                            params.image_features_dir,
                            self.entity_sent,
                            self.entity_bbox)
        self.phrases_data_loader = torch.utils.data.DataLoader(cd2,
                                                               batch_size=1,
                                                               collate_fn=cd2.collate,
                                                               shuffle=False, **kwargs)
