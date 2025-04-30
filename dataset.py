import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os, sys, ujson, gzip, cv2
from tqdm import tqdm
import numpy as np

class CustomDataset(Dataset):
  def __init__(self,
               root,
               skip_first = 5,
               seq_len = 1,
               rank=0):
    self.seq_len = seq_len
    self.img_h = 320
    self.img_w = 180
    self.images = []
    self.semantics = []
    self.bev_semantics = []
    self.depth = []
    self.cam_name = ["front_left", "front_right", "front_wide", "rear_left", "rear_right", "rear"] 
    self.rgb_name = ["rgb_"+i for i in self.cam_name] 
    self.semantics_name = ["semantic_"+i for i in self.cam_name] 
    self.depth_name = ["depth_"+i for i in self.cam_name]
    total_routes = 0
    crashed_routes = 0

    for sub_root in tqdm(root, file=sys.stdout, disable=rank != 0):

      # list subdirectories in root
    #   routes = next(os.walk(sub_root))[1]

      routes = sub_root
      for route in routes:
        route_dir = sub_root + '/' + route

        if not os.path.isfile(route_dir + '/results.json.gz'):
          total_routes += 1
          crashed_routes += 1
          continue

        with gzip.open(route_dir + '/results.json.gz', 'rt', encoding='utf-8') as f:
          total_routes += 1
          results_route = ujson.load(f)

        # We skip data where the expert did not achieve perfect driving score
        if results_route['scores']['score_composed'] < 100.0:
          pass

        num_seq = len(os.listdir(route_dir + '/rgb'))

        for seq in range(skip_first, num_seq - self.seq_len):
          # load input seq and pred seq jointly
          image = []
          semantic = []
          bev_semantic = []
          depth = []

          # Loads the current (and past) frames (if seq_len > 1)
          for idx in range(self.seq_len):
              image_i = []
              semantic_i = []
              depth_i = []
              for i in range(len(self.cam_name)):
                image_i.append(route_dir[:-1] + self.rgb_name[i] + (f'/{(seq + idx):04}.jpg'))
                semantic_i.append(route_dir + self.semantics_name[i] + (f'/{(seq + idx):04}.png'))
                depth_i.append(route_dir + self.depth_name[i] + (f'/{(seq + idx):04}.png'))
                
              bev_semantic.append(route_dir + '/bev_semantics' + (f'/{(seq + idx):04}.png'))
              image.append(image_i)
              semantic.append(semantic_i)
              depth.append(depth_i)
          self.images.append(image)
          self.semantics.append(semantic)
          self.bev_semantics.append(bev_semantic)
          self.depth.append(depth)

    # There is a complex "memory leak"/performance issue when using Python
    # objects like lists in a Dataloader that is loaded with
    # multiprocessing, num_workers > 0
    # A summary of that ongoing discussion can be found here
    # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
    # A workaround is to store the string lists as numpy byte objects
    # because they only have 1 refcount.
    self.images = np.array(self.images).astype(np.bytes_)
    self.semantics = np.array(self.semantics).astype(np.bytes_)
    self.bev_semantics = np.array(self.bev_semantics).astype(np.bytes_)
    self.depth = np.array(self.depth).astype(np.bytes_)

    self.perspective_downsample_factor = 1
    self.converter = [
        0,  # unlabeled
        0,  # building
        0,  # fence
        0,  # other
        4,  # pedestrian
        0,  # pole
        5,  # road line
        2,  # road
        6,  # sidewalk
        0,  # vegetation
        1,  # vehicle
        0,  # wall
        0,  # traffic sign
        0,  # sky
        0,  # ground
        0,  # bridge
        0,  # rail track
        0,  # guard rail
        3,  # traffic light
        0,  # static
        0,  # dynamic
        0,  # water
        0,  # terrain
    ]

    self.bev_converter = [
        0,  # unlabeled
        1,  # road
        2,  # sidewalk
        3,  # lane_markers
        4,  # lane_markers broken, you may cross them
        5,  # stop_signs
        6,  # traffic light green
        7,  # traffic light yellow
        8,  # traffic light red
        9,  # vehicle
        10,  # walker
    ]
    self.converter = np.uint8(self.converter)
    self.bev_converter = np.uint8(self.bev_converter)
    if rank == 0:
      print(f'Loading {len(self.images)} images from {len(root)} folders')
      print('Total amount of routes:', total_routes)
      print('Crashed routes:', crashed_routes)

  def __len__(self):
    """Returns the length of the dataset. """
    return self.images.shape[0]

  def __getitem__(self, idx):
    data = {}
    images = self.images[idx]
    semantics = self.semantics[idx]
    bev_semantics = self.bev_semantics[idx]
    depth = self.depth[idx]
    images_i = []
    semantics_i = []
    depth_i = []
    bev_semantics_i = []
    for i in range(self.seq_len):
        mimages_list = []
        mdepth_list = []
        msemantics_list = []
        for j in range(len(self.cam_name)):
            mimages_i = cv2.imread(str(images[i][j], encoding='utf-8'), cv2.IMREAD_COLOR)
            mimages_i = cv2.resize(mimages_i, (self.img_h, self.img_w), interpolation=cv2.INTER_AREA)
            mimages_i = cv2.cvtColor(mimages_i, cv2.COLOR_BGR2RGB)
            mimages_i = np.transpose(mimages_i, (2, 0, 1))
            mimages_list.append(mimages_i)

            msemantics_i = cv2.imread(str(semantics[i][j], encoding='utf-8'), cv2.IMREAD_UNCHANGED)
            msemantics_i = cv2.resize(msemantics_i, (self.img_h, self.img_w), interpolation=cv2.INTER_AREA)
            msemantics_i = self.converter[msemantics_i]
            msemantics_i = msemantics_i[::self.perspective_downsample_factor, ::self.perspective_downsample_factor]
            msemantics_list.append(msemantics_i)   

            mdepth_i = cv2.imread(str(depth[i][j], encoding='utf-8'), cv2.IMREAD_UNCHANGED)
            mdepth_i = cv2.resize(mdepth_i, (self.img_h, self.img_w), interpolation=cv2.INTER_AREA)
            mdepth_i = cv2.resize(mdepth_i,
                        dsize=(mdepth_i.shape[1] // self.perspective_downsample_factor,
                                mdepth_i.shape[0] // self.perspective_downsample_factor),
                        interpolation=cv2.INTER_LINEAR)
            mdepth_i = mdepth_i.astype(np.float32) / 255.0
            mdepth_list.append(mdepth_i)
        mbev_semantics_i = cv2.imread(str(bev_semantics[i], encoding='utf-8'), cv2.IMREAD_UNCHANGED)
        mbev_semantics_i = self.bev_converter[mbev_semantics_i]     
        images_i.append(mimages_list)
        semantics_i.append(msemantics_list)
        depth_i.append(mdepth_list)
        bev_semantics_i.append(mbev_semantics_i)
    data['images'] = images_i
    data['semantics'] = semantics_i
    data['depth'] = depth_i
    data['bev_semantics'] = bev_semantics_i
    return data

if __name__ == "__main__":
    root_dir = "/media/czg/DriveLab_Datastorage/geely_data(failed)"
    train_towns = os.listdir(root_dir)  # Scenario Folders
    first_val_town = 'Town02'
    second_val_town = 'Town05'
    val_towns = train_towns
    train_data, val_data = [], []
    for town in train_towns:
      root_files = os.listdir(os.path.join(root_dir, town))  # Town folders
      for file in root_files:
        if not os.path.isdir(os.path.join(root_dir, town, file)):
          continue
        # Only load as many repetitions as specified
        # repetition = int(re.search('Repetition(\\d+)', file).group(1))
        # if repetition >= num_repetitions:
        #   continue
        # We don't train on two towns and reserve them for validation
        if ((file.find(first_val_town) != -1) or (file.find(second_val_town) != -1)):
          continue
        if not os.path.isfile(os.path.join(root_dir, file)):
          train_data.append(os.path.join(root_dir, town, file))
    for town in val_towns:
      root_files = os.listdir(os.path.join(root_dir, town))
      for file in root_files:
        # repetition = int(re.search('Repetition(\\d+)', file).group(1))
        # if repetition >= num_repetitions:
        #   continue
        # Only use withheld towns for validation
        if ((file.find(first_val_town) == -1) and (file.find(second_val_town) == -1)):
          continue
        if not os.path.isfile(os.path.join(root_dir, file)):
          val_data.append(os.path.join(root_dir, town, file))
    # 创建Dataset和DataLoader
    dataset = CustomDataset(root=train_data)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

    # 使用示例
    for data in dataloader:
      print(data)
