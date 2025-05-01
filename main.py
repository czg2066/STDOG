import torch, os, wandb, cv2
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import CustomDataset
from model import CustomModel
from tqdm import tqdm
import numpy as np
# front_row = torch.cat([rgb_front_left, rgb_front, rgb_front_right], dim=3)
# back_row = torch.cat([rgb_back_left, rgb_back, rgb_back_right], dim=3)
# six_view_combined = torch.cat([front_row, back_row], dim=2)

class CustomTrain():
    def __init__(self, save_path, batch_size, resume, describe):
        self.save_path = save_path
        self.batch_size = batch_size
        self.resume = resume
        self.describe = describe

        self.all_step = 0
        self.epoch = 0

        self.model = CustomModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if resume: self.resume_model(save_path)
        self.model = torch.nn.DataParallel(self.model).to(self.device)
        wandb.init(project="Dog", name=save_path.split('/')[-2], notes=describe, id=save_path.split('/')[-2], resume="allow")

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
                images = data['images'][0]
                semantic_labels = data['semantics'][0]
                depth_labels = data['depth'][0]
                front_row = torch.cat([images[0], images[2], images[1]], dim=3)
                back_row = torch.cat([images[3], images[5], images[4]], dim=3)
                six_view_rgb = torch.cat([front_row, back_row], dim=2)
                front_row = torch.cat([depth_labels[0], depth_labels[2], depth_labels[1]], dim=-1)
                back_row = torch.cat([depth_labels[3], depth_labels[5], depth_labels[4]], dim=-1)
                six_view_depth = torch.cat([front_row, back_row], dim=-2)
                front_row = torch.cat([semantic_labels[0], semantic_labels[2], semantic_labels[1]], dim=-1)
                back_row = torch.cat([semantic_labels[3], semantic_labels[5], semantic_labels[4]], dim=-1)
                six_view_semantics = torch.cat([front_row, back_row], dim=-2)
                six_view_rgb = six_view_rgb.to(self.device, torch.float32)
                six_view_semantics = six_view_semantics.to(self.device, torch.long)
                six_view_depth = six_view_depth.to(self.device, torch.float32)
                bev_semantics = data['bev_semantics'][0].to(self.device, torch.long)
                semantic_features, depth_features, bev_features = self.model(six_view_rgb)
                loss = self.model.module.loss_all(semantic_features, depth_features, bev_features, six_view_semantics, six_view_depth, bev_semantics)
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
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
                if self.all_step % ((int(len(dataset)/self.batch_size)+1)//10) == 0:
                    wandb.log({
                        "epoch": self.epoch,
                        "step": self.all_step,
                        "loss": total_loss.item(),
                        "semloss": loss[0].item(),
                        "deploss": loss[1].item(),
                        "bevloss": loss[2].item(),
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
        with torch.no_grad():
            for batch_idx, data in progress:
                images = data['images'][0]
                semantic_labels = data['semantics'][0]
                depth_labels = data['depth'][0]
                front_row = torch.cat([depth_labels[0], depth_labels[2], depth_labels[1]], dim=-1)
                back_row = torch.cat([depth_labels[3], depth_labels[5], depth_labels[4]], dim=-1)
                six_view_depth = torch.cat([front_row, back_row], dim=-2)
                front_row = torch.cat([semantic_labels[0], semantic_labels[2], semantic_labels[1]], dim=-1)
                back_row = torch.cat([semantic_labels[3], semantic_labels[5], semantic_labels[4]], dim=-1)
                six_view_semantics = torch.cat([front_row, back_row], dim=-2)
                # for i in range(len(images)):
                #     images[i] = images[i].to(device, torch.float32)
                front_row = torch.cat([images[0], images[2], images[1]], dim=3)
                back_row = torch.cat([images[3], images[5], images[4]], dim=3)
                six_view_rgb = torch.cat([front_row, back_row], dim=2)
                six_view_rgb = six_view_rgb.to(self.device, torch.float32)
                six_view_semantics = six_view_semantics.to(self.device, torch.long)
                six_view_depth = six_view_depth.to(self.device, torch.float32)
                bev_semantics = data['bev_semantics'][0].to(self.device, torch.long)
                semantic_features, depth_features, bev_features = self.model(six_view_rgb)
                loss = self.model.module.loss_all(semantic_features, depth_features, bev_features, six_view_semantics, six_view_depth, bev_semantics)
                total_loss = torch.sum(torch.stack(loss))
                # 实时更新进度条信息
                progress.set_postfix({
                    'losses': f"{total_loss.item():.4f}",
                    'semloss': f"{loss[0].item():.4f}",
                    'deploss': f"{loss[1].item():.4f}",
                    'bevloss': f"{loss[2].item():.4f}"
                })
                total_losses.append(total_loss.item())
                semlosses.append(loss[0].item())
                deplosses.append(loss[1].item())
                bevlosses.append(loss[2].item())
                if bev_features is not None:
                    bev_features = torch.argmax(bev_features, dim=1)
                    semantic_features = torch.argmax(semantic_features, dim=1)
                    ba = 0
                    for i in range(len(bev_features)):
                        bev_semantic_indices = bev_features[i].detach().cpu().numpy()
                        rgb_feature = six_view_rgb[i].permute(1, 2, 0).detach().cpu().numpy()
                        bev_semantic = bev_semantics[i].detach().cpu().numpy()
                        semantic_feature = semantic_features[i].detach().cpu().numpy()
                        
                        bev_converter = np.array(bev_classes_list)
                        converter = np.array(classes_list)
                        bev_converter[1][0:3] = 40
                        bev_semantic_image = bev_converter[bev_semantic_indices, ...].astype('uint8')
                        bev_semantic_label = bev_converter[bev_semantic, ...].astype('uint8')
                        semantic_image = converter[semantic_feature, ...].astype('uint8')
                        cv2.imwrite(self.save_path+"plots/"+f'{batch_idx}_{ba}_bev_semantic.png', bev_semantic_label)
                        cv2.imwrite(self.save_path+"plots/"+f'{batch_idx}_{ba}_bev.png', bev_semantic_image)
                        cv2.imwrite(self.save_path+"plots/"+f'{batch_idx}_{ba}_semantic.png', semantic_image)
                        
                        cv2.imwrite(self.save_path+"plots/"+f'{batch_idx}_{ba}_rgb.png', rgb_feature)
                        ba += 1
        print(f"Test finished.")
        print(f"Total loss: {sum(total_losses)/len(total_losses)}")
        print(f"Semantic loss: {sum(semlosses)/len(semlosses)}")
        print(f"Depth loss: {sum(deplosses)/len(deplosses)}")
        print(f"BEV loss: {sum(bevlosses)/len(bevlosses)}")


if __name__ == "__main__":
    root_dir = "/media/czg/DriveLab_Datastorage/geely_data_radar_faild"
    id = "v11_0501"
    describe="v10.2出错，怀疑是新家的多头注意力维度设置导致的问题，重新训练"
    save_path = "./trained_model/"+id+"/"
    os.makedirs(os.path.dirname(save_path), exist_ok=True) 
    os.makedirs(os.path.dirname(save_path+"plots/"), exist_ok=True)
    epochs = 30
    loop_mun = 10
    batch_size = 10
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
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=32)
    val_dataset = CustomDataset(root=val_data)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=32)
    own_train = CustomTrain(save_path, batch_size, resume, describe)
    
    for _ in range(loop_mun):
       own_train.model_traiin(val_dataloader, val_dataset, epochs=epochs//loop_mun)
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
