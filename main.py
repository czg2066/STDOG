import torch, os, wandb, cv2
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import CustomDataset
from model_st import CustomModel
from tqdm import tqdm
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
import torch.backends.cudnn as cudnn

DDP_or_DP = "DP"

class CustomTrain():
    def __init__(self, save_path, batch_size, resume, describe, rank):
        self.save_path = save_path
        self.batch_size = batch_size
        self.resume = resume
        self.describe = describe

        self.all_step = 0
        self.epoch = 0
        self.rank = rank
        if DDP_or_DP == "DDP":
            self.device = torch.device("cuda", self.rank)
        else:
            self.device = torch.device("cuda")

        self.model = CustomModel().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        if resume: self.resume_model(save_path)
        if DDP_or_DP == "DDP":
            self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank, find_unused_parameters=True)
        else:
            self.model = nn.DataParallel(self.model)
        wandb.init(project="Dog_new", name=save_path.split('/')[-2], notes=describe, id=save_path.split('/')[-2], resume="allow")

    def resume_model(self, save_path):
        files = os.listdir(save_path)
        if len(files) == 0:
            print("No checkpoint found! Retrain from scratch!")
        else:
            pth_files = [file for file in files if file.endswith('.pth')]
            pth_files.sort(key=lambda x:int(x[11:-4]))
            print(f"Loading checkpoint from {save_path+pth_files[-1]}")
            checkpoint = torch.load(save_path+pth_files[-1])
            # 加载参数（自动匹配多GPU参数名称）
            self.model.load_state_dict(checkpoint['model_state_dict'])
            # 恢复训练状态（可选）
            self.epoch = checkpoint['epoch']+1
            print(f"Resume from epoch {checkpoint['epoch']}, set epoch to {self.epoch}")
            self.all_step = checkpoint['all_step']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            print("Resume success!")
       

    def model_traiin(self, dataloader, dataset, epochs=5):
        self.model.train()
        epochs += self.epoch
        while self.epoch < epochs:
            progress = tqdm(
                enumerate(dataloader), 
                total=int(len(dataset)/self.batch_size)+1,
                desc=f"Epoch {self.epoch+1}/{epochs}",
                bar_format="{l_bar}{bar:30}{r_bar}",
                ncols=180  # 控制进度条宽度
            )
            # 使用示例
            for batch_idx, data in progress:
                images = data['images'].squeeze(1)
                semantic_labels = data['semantics'].squeeze(1)
                depth_labels = data['depth'].squeeze(1)
                bev_semantics = data['bev_semantics'].squeeze(1)
                ego_waypoints = data['ego_waypoints']
                front_row = torch.cat([images[:, 0], images[:, 2], images[:, 1]], dim=3)
                back_row = torch.cat([images[:, 3], images[:, 5], images[:, 4]], dim=3)
                # dim=2 (高度维度) 用于拼接前后两排视图
                six_view_rgb = torch.cat([front_row, back_row], dim=2)
                front_row = torch.cat([depth_labels[:, 0], depth_labels[:, 2], depth_labels[:, 1]], dim=-1)
                back_row = torch.cat([depth_labels[:, 3], depth_labels[:, 5], depth_labels[:, 4]], dim=-1)
                # dim=-2 (倒数第二个维度，即高度)
                six_view_depth = torch.cat([front_row, back_row], dim=-2)
                front_row = torch.cat([semantic_labels[:, 0], semantic_labels[:, 2], semantic_labels[:, 1]], dim=-1)
                back_row = torch.cat([semantic_labels[:, 3], semantic_labels[:, 5], semantic_labels[:, 4]], dim=-1)
                six_view_semantics = torch.cat([front_row, back_row], dim=-2)

                six_view_rgb = six_view_rgb.contiguous()
                six_view_semantics = six_view_semantics.contiguous()
                six_view_depth = six_view_depth.contiguous()

                six_view_rgb = six_view_rgb.to(self.device, torch.float32)
                six_view_semantics = six_view_semantics.to(self.device, torch.long)
                six_view_depth = six_view_depth.to(self.device, torch.float32)
                bev_semantics = bev_semantics.to(self.device, torch.long)
                ego_waypoints = ego_waypoints.to(self.device, torch.float32)
                semantic_features, depth_features, bev_features, trajectory_features = self.model(six_view_rgb)
                loss = self.model.module.loss_all(semantic_features, depth_features, bev_features, trajectory_features, six_view_semantics, six_view_depth, bev_semantics, ego_waypoints)
                self.optimizer.zero_grad()
                total_loss = torch.sum(torch.stack(loss))
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                # 实时更新进度条信息
                progress.set_postfix({
                    'losses': f"{total_loss.item():.4f}",
                    'semloss': f"{loss[0].item():.4f}",
                    'deploss': f"{loss[1].item():.4f}",
                    'bevloss': f"{loss[2].item():.4f}",
                    'trajloss': f"{loss[3].item():.4f}" if len(loss) >= 4 else "None",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
                if self.all_step % ((int(len(dataset)/self.batch_size)+1)//100) == 0:
                    wandb.log({
                        "epoch": self.epoch,
                        "step": self.all_step,
                        "loss": total_loss.item(),
                        "semloss": loss[0].item(),
                        "deploss": loss[1].item(),
                        "bevloss": loss[2].item(),
                        'trajloss': loss[3].item() if len(loss) >= 4 else "None",
                        "lr": self.optimizer.param_groups[0]['lr']
                    })
                self.all_step += 1
            
            torch.save({
                'epoch': self.epoch,
                'all_step': self.all_step,
                'model_state_dict': self.model.module.state_dict(),  # 关键：通过module获取原始模型
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
            }, self.save_path+f'model_epoch{self.epoch}.pth')
            print(f"Epoch {self.epoch+1}/{epochs} finished.")
            self.epoch += 1

    def model_test(self, dataloader, dataset):
        self.model.eval()
        progress = tqdm(
            enumerate(dataloader), 
            total=int(len(dataset)/self.batch_size)+1, 
            desc=f"Test",
            bar_format="{l_bar}{bar:30}{r_bar}",
            ncols=180  # 控制进度条宽度
        )
        # 使用示例
        total_losses = []
        semlosses = []
        deplosses = []
        bevlosses = []
        trajlosses = []
        bev_classes_list = [
            [0, 0, 0],  # unlabeled
            [200, 200, 200],  # road
            [255, 255, 255],  # sidewalk
            [255, 255, 0],  # road line
            [50, 234, 157],  # road line broken
            [160, 160, 0],  # stop sign
            [0, 255, 0],  # light green
            [255, 255, 0],  # light yellow
            [255, 0, 0],  # light red
            [250, 170, 30],  # vehicle
            [0, 255, 0],  # pedestrian
        ]
        classes_list = [
            [0, 0, 0],  # unlabeled
            [250, 170, 30],  # vehicle
            [200, 200, 200],  # road
            [0, 255, 255],  # light
            [0, 255, 0],  # pedestrian
            [255, 255, 0],  # road line
            [255, 255, 255],  # sidewalk
        ]
        bev_converter = np.array(bev_classes_list, dtype=np.uint8)
        converter = np.array(classes_list, dtype=np.uint8)
        bev_converter[1, :3] = 40

        with torch.no_grad():
            for batch_idx, data in progress:
                images = data['images'].squeeze(1)
                semantic_labels = data['semantics'].squeeze(1)
                depth_labels = data['depth'].squeeze(1)
                bev_semantics = data['bev_semantics'].squeeze(1)
                ego_waypoints = data['ego_waypoints'].squeeze(1)
                front_row = torch.cat([images[:, 0], images[:, 2], images[:, 1]], dim=3)
                back_row = torch.cat([images[:, 3], images[:, 5], images[:, 4]], dim=3)
                # dim=2 (高度维度) 用于拼接前后两排视图
                six_view_rgb = torch.cat([front_row, back_row], dim=2)
                front_row = torch.cat([depth_labels[:, 0], depth_labels[:, 2], depth_labels[:, 1]], dim=-1)
                back_row = torch.cat([depth_labels[:, 3], depth_labels[:, 5], depth_labels[:, 4]], dim=-1)
                # dim=-2 (倒数第二个维度，即高度)
                six_view_depth = torch.cat([front_row, back_row], dim=-2)
                front_row = torch.cat([semantic_labels[:, 0], semantic_labels[:, 2], semantic_labels[:, 1]], dim=-1)
                back_row = torch.cat([semantic_labels[:, 3], semantic_labels[:, 5], semantic_labels[:, 4]], dim=-1)
                six_view_semantics = torch.cat([front_row, back_row], dim=-2)
                six_view_rgb = six_view_rgb.to(self.device, torch.float32)
                six_view_semantics = six_view_semantics.to(self.device, torch.long)
                six_view_depth = six_view_depth.to(self.device, torch.float32)
                bev_semantics = bev_semantics.to(self.device, torch.long)
                ego_waypoints = ego_waypoints.to(self.device, torch.float32)
                semantic_features, depth_features, bev_features, trajectory_features = self.model(six_view_rgb)
                loss = self.model.module.loss_all(semantic_features, depth_features, bev_features, trajectory_features, six_view_semantics, six_view_depth, bev_semantics, ego_waypoints)
                total_loss = torch.sum(torch.stack(loss))
                # 实时更新进度条信息
                progress.set_postfix({
                    'losses': f"{total_loss.item():.4f}",
                    'semloss': f"{loss[0].item():.4f}",
                    'deploss': f"{loss[1].item():.4f}",
                    'bevloss': f"{loss[2].item():.4f}",
                    'trajloss': f"{loss[3].item():.4f}"
                })
                total_losses.append(total_loss.item())
                semlosses.append(loss[0].item())
                deplosses.append(loss[1].item())
                bevlosses.append(loss[2].item())
                trajlosses.append(loss[3].item())
                if bev_features is not None and self.rank == 0 and batch_idx%10 == 0:
                    bev_features = torch.argmax(bev_features, dim=1)
                    semantic_features = torch.argmax(semantic_features, dim=1)
                    ba = 0
                    for i in range(len(bev_features)):
                        bev_semantic_indices = bev_features[i].detach().cpu().numpy()
                        rgb_feature = six_view_rgb[i].permute(1, 2, 0).detach().cpu().numpy()
                        bev_semantic = bev_semantics[i].detach().cpu().numpy()
                        semantic_feature = semantic_features[i].detach().cpu().numpy()
                        
                        bev_semantic_image = bev_converter[bev_semantic_indices, ...]
                        bev_semantic_label = bev_converter[bev_semantic, ...]
                        semantic_image = converter[semantic_feature, ...]
                        
                        Image.fromarray(bev_semantic_label).save(self.save_path+"plots/"+f'{batch_idx}_{ba}_bev_semantic.png')
                        Image.fromarray(bev_semantic_image).save(self.save_path+"plots/"+f'{batch_idx}_{ba}_bev.png')
                        Image.fromarray(semantic_image).save(self.save_path+"plots/"+f'{batch_idx}_{ba}_semantic.png')
                        
                        rgb_image_uint8 = (rgb_feature * 255).astype(np.uint8)
                        Image.fromarray(rgb_image_uint8).save(self.save_path+"plots/"+f'{batch_idx}_{ba}_rgb.png')
                        ba += 1
        print(f"Test finished.")
        print(f"Total loss: {sum(total_losses)/len(total_losses)}")
        print(f"Semantic loss: {sum(semlosses)/len(semlosses)}")
        print(f"Depth loss: {sum(deplosses)/len(deplosses)}")
        print(f"BEV loss: {sum(bevlosses)/len(bevlosses)}")
        print(f"Trajectory loss: {sum(trajlosses)/len(trajlosses)}")

if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    if DDP_or_DP == "DDP":
        rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="gloo")
        torch.cuda.set_device(rank)
        print(f"[Rank {dist.get_rank()}] Initialization finished. Using GPU: {rank}")
    else:
        rank = None
    root_dir = "/media/zr/project/STDOG/data"
    id = "v0_719_1"
    describe="测试带参考点的时空注意力,双3090"
    save_path = "./trained_model/"+id+"/"
    os.makedirs(os.path.dirname(save_path), exist_ok=True) 
    os.makedirs(os.path.dirname(save_path+"plots/"), exist_ok=True)
    epochs = 30
    loop_mun = 10
    batch_size = 24
    num_workers = 16
    resume = False
    train_towns = os.listdir(root_dir)  # Scenario FoldersFalse
    first_val_town = 'Town02'
    second_val_town = 'Town05'
    val_towns = train_towns
    train_data, val_data = [], []
    for town in train_towns:
      root_files = os.listdir(os.path.join(root_dir, town))  # Town folders
      for file in root_files:
        if not os.path.isdir(os.path.join(root_dir, town, file)):
          continue
        if ((file.find(first_val_town) != -1) or (file.find(second_val_town) != -1)):
          continue
        if not os.path.isfile(os.path.join(root_dir, file)):
          train_data.append(os.path.join(root_dir, town, file))
    for town in val_towns:
      root_files = os.listdir(os.path.join(root_dir, town))
      for file in root_files:
        if ((file.find(first_val_town) == -1) and (file.find(second_val_town) == -1)):
          continue
        if not os.path.isfile(os.path.join(root_dir, file)):
          val_data.append(os.path.join(root_dir, town, file))
    # 创建Dataset和DataLoader
    train_dataset = CustomDataset(root=train_data)
    if DDP_or_DP == "DDP":
        train_sampler = DistributedSampler(train_dataset, rank=rank)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_dataset = CustomDataset(root=val_data)
    if DDP_or_DP == "DDP":
        val_sampler = DistributedSampler(val_dataset, rank=rank)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    else:
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    own_train = CustomTrain(save_path, batch_size, resume, describe, rank=rank)
    
    for _ in range(loop_mun):
       own_train.model_traiin(train_dataloader, train_dataset, epochs=epochs//loop_mun)
       own_train.model_test(val_dataloader, val_dataset)
        # model_traiin(save_path, val_dataloader, val_dataset, epochs=5, batch_size=batch_size, resume=resume, \
        #              describe="loss一开始健康但是后面崩了，增加IL损失，降低梯度裁减，重新训练")
        # model_test(save_path, val_dataloader, val_dataset, batch_size)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = CustomModel()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # # wandb.init(project="Dog", name="v0_0424", notes="首次尝试视角侧抑制注意力和损失以及高斯差分函数(DOG)", resume=True)
    # all_step = 0
    # epoch = 0
    # if resume:
    #     checkpoint = torch.load('trained_model/model_epoch{epoch}.pth')
    #     # 加载参数（自动匹配多GPU参数名称）
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     # 恢复训练状态（可选）
    #     epoch = checkpoint['epoch']
    #     all_step = checkpoint['all_step']
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # model = nn.DataParallel(model)
    # model = model.to(device)
    # while epoch < epochs:
    #     progress = tqdm(
    #         enumerate(train_dataloader), 
    #         total=int(len(train_dataset)/batch_size)+1,
    #         desc=f"Epoch {epoch+1}/{epochs}",
    #         bar_format="{l_bar}{bar:30}{r_bar}",
    #         ncols=180  # 控制进度条宽度
    #     )
    #     # 使用示例
    #     for batch_idx, data in progress:
    #         images = data['images'][0]
    #         semantic_labels = data['semantics'][0]
    #         depth_labels = data['depth'][0]
    #         front_row = torch.cat([images[0], images[2], images[1]], dim=3)
    #         back_row = torch.cat([images[3], images[5], images[4]], dim=3)
    #         six_view_rgb = torch.cat([front_row, back_row], dim=2)
    #         front_row = torch.cat([depth_labels[0], depth_labels[2], depth_labels[1]], dim=-1)
    #         back_row = torch.cat([depth_labels[3], depth_labels[5], depth_labels[4]], dim=-1)
    #         six_view_depth = torch.cat([front_row, back_row], dim=-2)
    #         front_row = torch.cat([semantic_labels[0], semantic_labels[2], semantic_labels[1]], dim=-1)
    #         back_row = torch.cat([semantic_labels[3], semantic_labels[5], semantic_labels[4]], dim=-1)
    #         six_view_semantics = torch.cat([front_row, back_row], dim=-2)
    #         six_view_rgb = six_view_rgb.to(device, torch.float32)
    #         six_view_semantics = six_view_semantics.to(device, torch.long)
    #         six_view_depth = six_view_depth.to(device, torch.float32)
    #         bev_semantics = data['bev_semantics'][0].to(device, torch.long)
    #         semantic_features, depth_features, bev_features = model(six_view_rgb)
    #         loss = model.module.loss_all(semantic_features, depth_features, bev_features, six_view_semantics, six_view_depth, bev_semantics)
    #         optimizer.zero_grad()
    #         total_loss = torch.sum(torch.stack(loss))
    #         total_loss.backward()
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), 15.0)
    #         optimizer.step()
    #         # 实时更新进度条信息
    #         progress.set_postfix({
    #             'losses': f"{total_loss.item():.4f}",
    #             'semloss': f"{loss[0].item():.4f}",
    #             'deploss': f"{loss[1].item():.4f}",
    #             'bevloss': f"{loss[2].item():.4f}",
    #             'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
    #         })
    #         # if all_step % 10 == 0:
    #         #     wandb.log({
    #         #         "epoch": epoch,
    #         #         "step": all_step,
    #         #         "loss": total_loss.item(),
    #         #         "semloss": loss[0].item(),
    #         #         "deploss": loss[1].item(),
    #         #         "bevloss": loss[2].item(),
    #         #         "lr": optimizer.param_groups[0]['lr']
    #         #     })
    #         all_step += 1
        
    #     torch.save({
    #         'epoch': epoch,
    #         'all_step': all_step,
    #         'model_state_dict': model.module.state_dict(),  # 关键：通过module获取原始模型
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'loss': loss,
    #     }, save_path+f'model_epoch{epoch}.pth')
    #     print(f"Epoch {epoch+1}/{epochs} finished.")
    #     epoch += 1
