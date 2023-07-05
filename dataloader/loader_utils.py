import h5py
import torch

class ImageFeaturesHdfReader(object):
    """
    A reader for HDF files containing pre-extracted image features. A typical
    HDF file is expected to have a column named "image_id", and another column
    named "features".
    Example of an HDF file:
    ```
    visdial_train_faster_rcnn_bottomup_features.h5
       |--- "image_id" [shape: (num_images, )]
       |--- "features" [shape: (num_images, num_proposals, feature_size)]
       +--- .attrs ("split", "train")
    ```
    Refer ``$PROJECT_ROOT/data/extract_bottomup.py`` script for more details
    about HDF structure.
    Parameters
    ----------
    features_hdfpath : str
        Path to an HDF file containing VisDial v1.0 train, val or test split
        image features.
    in_memory : bool
        Whether to load the whole HDF file in memory. Beware, these files are
        sometimes tens of GBs in size. Set this to true if you have sufficient
        RAM - trade-off between speed and memory.
    """

    def __init__(self, features_hdfpath: str, in_memory: bool = False):
        self.features_hdfpath = features_hdfpath
        self._in_memory = in_memory
        print(self.features_hdfpath)
        with h5py.File(self.features_hdfpath, 'r') as features_hdf:
            # self._split = features_hdf.attrs["split"]
            self._image_id_list = list(features_hdf["img_id"])
            # "features" is List[np.ndarray] if the dataset is loaded in-memory
            # If not loaded in memory, then list of None.
            self.features = [None] * len(self._image_id_list)
            self.boxes = [None] * len(self._image_id_list)
            self.classes = [None] * len(self._image_id_list)
            self.scores = [None] * len(self._image_id_list)
            self.img_ws = [None] * len(self._image_id_list)
            self.img_hs = [None] * len(self._image_id_list)
            self.num_boxes_list = [None] * len(self._image_id_list)

    def __len__(self):
        return len(self._image_id_list)

    def __getitem__(self, image_id: int):
        index = self._image_id_list.index(image_id)
        if self._in_memory:
            # Load features during first epoch, all not loaded together as it
            # has a slow start.
            if self.features[index] is not None:
                image_id_features = self.features[index]
                boxes = self.boxes[index]
                single_class = self.classes[index]
                single_score = self.scores[index]
                img_w = self.img_ws[index]
                img_h = self.img_hs[index]
                num_boxes = self.num_boxes_list[index]

            else:
                with h5py.File(self.features_hdfpath, "r") as features_hdf:
                    image_id_features = features_hdf["features"][index]
                    boxes = features_hdf["boxes"][index]
                    single_class = features_hdf["objects_id"][index]
                    single_score = features_hdf["objects_conf"][index]
                    img_w = features_hdf['img_w'][index]
                    img_h = features_hdf['img_h'][index]
                    num_boxes = features_hdf['num_boxes'][index]

                    self.features[index] = image_id_features
                    self.boxes[index] = boxes
                    self.classes[index] = single_class
                    self.scores[index] = single_score
                    self.img_ws[index] = img_w
                    self.img_hs[index] = img_h
                    self.num_boxes_list[index] = num_boxes
 
        else:
            # Read chunk from file everytime if not loaded in memory.
            with h5py.File(self.features_hdfpath, "r") as features_hdf:
                image_id_features = features_hdf["features"][index]
                boxes = features_hdf["boxes"][index]
                single_class = features_hdf["objects_id"][index]
                single_score = features_hdf["objects_conf"][index]
                img_w = features_hdf['img_w'][index]
                img_h = features_hdf['img_h'][index]
                num_boxes = features_hdf['num_boxes'][index]
        
        boxes = torch.from_numpy(boxes)
        rel_boxes = boxes.clone()
        rel_boxes[:, [0, 2]] /= img_w
        rel_boxes[:, [1, 3]] /= img_h

        visual_attention_mask = (torch.arange(boxes.shape[0]) < num_boxes).long()

        return {
            'features': torch.from_numpy(image_id_features),
            'rel_boxes': rel_boxes,
            'visual_attention_mask': visual_attention_mask
        }
    
    def get_batch_img_info(self, img_id_list):
        batch = [self.__getitem__(i) for i in img_id_list]
        mergedBatch = {k: [d[k] for d in batch] for k in batch[0]}
        out = {}
        for key in mergedBatch:
            if isinstance(mergedBatch[key][0], int):
                out[key] = torch.tensor(mergedBatch[key])
            else:
                out[key] = torch.stack(mergedBatch[key])
        return out

    # # (36, 2053)
    # def process_feat(self, img_feat, boxes, img_w, img_h):
    #     img_feat = torch.from_numpy(img_feat)
    #     boxes = torch.from_numpy(boxes)
    #     relateive_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    #     vis_pe = torch.cat([boxes, relateive_area.unsqueeze(1)], dim=1)
    #     vis_pe[:,[0,2,4]] /= img_w
    #     vis_pe[:, [1,3,4]] /= img_h # vis_pe: [x1 / w, y1 / h, x2 / w, y2 / h, (x2-x1)*(y2-y1) / (w*h)]
    #     img_feat = torch.cat([img_feat, vis_pe], dim=-1) # (batch_size, 36, 2053)
    #     return img_feat


def pad_sequence(x, max_sequence_len, y=0, dtype=torch.long):
    cur_sequence_len = x.size(0)
    constants = torch.zeros([max_sequence_len - cur_sequence_len] + list(x.size()[1:]), dtype=dtype)
    constants.fill_(y)
    x = torch.cat([x, constants], dim=0)
    return x

def pad_sequence_list(seq_list, y=0, dtype=torch.long):
    max_sequence_len = max([x.size(0) for x in seq_list])
    for i in range(len(seq_list)):
        seq_list[i] = pad_sequence(seq_list[i], max_sequence_len, y=y, dtype=dtype)
    ret = torch.stack(seq_list)
    return ret