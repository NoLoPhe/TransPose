import torch
import os
from config import paths
from utils import normalize_and_concat
from net import TransPoseNet
# import articulate as art

import torch
import os
from config import paths
import open3d as o3d
import articulate as art
import numpy as np
from tqdm import tqdm

from pygame.time import Clock

body_model_1 = art.ParametricModel(paths.smpl_file)
body_model_2 = art.ParametricModel(paths.smpl_file)
mesh = o3d.geometry.TriangleMesh()
vis = o3d.visualization.Visualizer()

class Main1:
    def __init__(self):
        print('hello')
        self.tran_list_first = None

    def view_motion(self, pose_list: list, tran_list: list, j: int, distance_between_subjects=0.8):
        verts = []
        if j == 0:
            self.tran_list_first = tran_list
        for i in range(len(pose_list)):
            pose = pose_list[i].view(-1, len(body_model_1.parent), 3, 3)
            tran = tran_list[i].view(-1, 3) - self.tran_list_first[i].view(-1, 3)[:1] if tran_list else None
            # tran = tran_list[i].view(-1, 3) - tran_list[i].view(-1, 3)[:1] if tran_list else None
            verts.append(body_model_1.forward_kinematics(pose, tran=tran, calc_mesh=True)[2])
        self.view_mesh(verts, j, distance_between_subjects=distance_between_subjects)

    def view_mesh(self, vertex_list: list, j:int, distance_between_subjects=0.8):
        f = body_model_1.face.copy()
        body_model_1.v_list = []
        body_model_1.f_list = []
        for i in range(len(vertex_list)):
            a1 = vertex_list[i]     # torch.Size([1, 6890, 3])
            # tensor([[[ 0.2457,  0.6446, -0.1137],
            #          [ 0.2458,  0.6289, -0.1070],
            #          [ 0.2348,  0.6297, -0.1169],
            #          ...,
            #          [ 0.1454,  0.6595,  0.0179],
            #          [ 0.1479,  0.6592,  0.0183],
            #          [ 0.1493,  0.6561,  0.0215]]])
            v = vertex_list[i].clone().view(-1, body_model_1._v_template.shape[0], 3)   # torch.Size([1, 6890, 3])
            # tensor([[[ 0.2457,  0.6446, -0.1137],
            #          [ 0.2458,  0.6289, -0.1070],
            #          [ 0.2348,  0.6297, -0.1169],
            #          ...,
            #          [ 0.1454,  0.6595,  0.0179],
            #          [ 0.1479,  0.6592,  0.0183],
            #          [ 0.1493,  0.6561,  0.0215]]])
            v[:, :, 0] += distance_between_subjects * i
            b1 = v      # torch.Size([1, 6890, 3])
            # tensor([[[ 0.2457,  0.6446, -0.1137],
            #          [ 0.2458,  0.6289, -0.1070],
            #          [ 0.2348,  0.6297, -0.1169],
            #          ...,
            #          [ 0.1454,  0.6595,  0.0179],
            #          [ 0.1479,  0.6592,  0.0183],
            #          [ 0.1493,  0.6561,  0.0215]]])
            body_model_1.v_list.append(v)
            body_model_1.f_list.append(f.copy())
            c1 = v.shape[1]
            f += v.shape[1]     # (13776, 3)
            # [[6891 6892 6890], [6890 6892 6893], [6892 6891 6894], [6894 6891 6895], [6892 6896 6893], [6893 6896 6897], [6899 6900 6898], [6898 6900 6901], [6903 6904 6902], [6902 6904 6905], [6907 6908 6906], [6906 6908 6909], [6911 6912 6910], [6910 6912 6913], [69
            cc1 = 0

        verts = torch.cat(body_model_1.v_list, dim=1).cpu().numpy()     # (1, 6890, 3)
        # [[[ 0.24571139  0.6446374  -0.1137421 ],  [ 0.24575582  0.6289439  -0.10696167],  [ 0.23476103  0.6297237  -0.11690235],  ...,  [ 0.14539975  0.6594848   0.01786305],  [ 0.14786917  0.6591695   0.01827737],  [ 0.1493457   0.6560608   0.02149733]]]
        faces = np.concatenate(body_model_1.f_list)     # (13776, 3)
        # [[  1   2   0], [  0   2   3], [  2   1   4], [  4   1   5], [  2   6   3], [  3   6   7], [  9  10   8], [  8  10  11], [ 13  14  12], [ 12  14  15], [ 17  18  16], [ 16  18  19], [ 21  22  20], [ 20  22  23], [ 25  17  24], [ 24  17  16], [ 26  27  16],
        d1 = 0

        self.render_sequence_3d(verts, faces, j, 1080, 1080, visible=True)

    def render_sequence_3d(self, verts, faces, j, width, height, visible=False, need_norm=True):
        if j == 0:
            mean = np.array([[[1.8154894, 0.06202411, -1.842172]]])
            # scale = 5.836794
            scale = 7
            verts = (verts - mean) / scale      # (1, 6890, 3)
            # [[[-0.27022413  0.10027727  0.29637225],  [-0.27019878  0.09759067  0.29752819],  [-0.27208537  0.09771936  0.29582778],  ...,  [-0.287387    0.1027794   0.31894609],  [-0.28696333  0.10272631  0.31901628],  [-0.28670822  0.10219505  0.3195682 ]]]
            # (978, 6890, 3)
            # [[[-0.27022413  0.10027727  0.29637225],  [-0.27019878  0.09759067  0.29752819],  [-0.27208537  0.09771936  0.29582778],  ...,  [-0.287387    0.1027794   0.31894609],  [-0.28696333  0.10272631  0.31901628],  [-0.28670822  0.10219505  0.3195682 ]],, [[-0.26
            e2 = verts[0]
            cam_offset = 1.2
            f2 = faces      # (13776, 3)
            # [[  1   2   0], [  0   2   3], [  2   1   4], [  4   1   5], [  2   6   3], [  3   6   7], [  9  10   8], [  8  10  11], [ 13  14  12], [ 12  14  15], [ 17  18  16], [ 16  18  19], [ 21  22  20], [ 20  22  23], [ 25  17  24], [ 24  17  16], [ 26  27  16],
            # (13776, 3)
            # [[  1   2   0], [  0   2   3], [  2   1   4], [  4   1   5], [  2   6   3], [  3   6   7], [  9  10   8], [  8  10  11], [ 13  14  12], [ 12  14  15], [ 17  18  16], [ 16  18  19], [ 21  22  20], [ 20  22  23], [ 25  17  24], [ 24  17  16], [ 26  27  16],
            # mesh = o3d.geometry.TriangleMesh()
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.vertices = o3d.utility.Vector3dVector(verts[0])

            # vis = o3d.visualization.Visualizer()
            vis.create_window(width=width, height=height, visible=visible)
            vis.add_geometry(mesh)
            view_control = vis.get_view_control()
            cam_params = view_control.convert_to_pinhole_camera_parameters()
            cam_params.extrinsic = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, cam_offset],
                [0, 0, 0, 1],
            ])
            view_control.convert_from_pinhole_camera_parameters(cam_params)

            mesh.vertices = o3d.utility.Vector3dVector(verts[0])
            m1 = verts[0]   # (6890, 3)
            # [[-0.27022413  0.10027727  0.29637225], [-0.27019878  0.09759067  0.29752819], [-0.27208537  0.09771936  0.29582778], [-0.27268847  0.10000165  0.29472659], [-0.27182195  0.09587249  0.29652361], [-0.27053997  0.09560653  0.29818329], [-0.27424189  0.09760
            mesh.compute_vertex_normals()
            vis.update_geometry(mesh)
            vis.poll_events()
            vis.update_renderer()
        else:
            mean = np.array([[[1.8154894, 0.06202411, -1.842172]]])
            scale = 5.836794
            verts = (verts - mean) / scale

            mesh.vertices = o3d.utility.Vector3dVector(verts[0])
            m2 = verts[0]
            # [[-0.26894525  0.09981735  0.29612659], [-0.26893764  0.09712863  0.29728826], [-0.27082134  0.09726223  0.29558515], [-0.27141733  0.09954588  0.29448379], [-0.27056641  0.09541444  0.29628263], [-0.26928803  0.09514545  0.29794438], [-0.27297654  0.09715


            # [[-0.26860178  0.0997606   0.29553324], [-0.26859822  0.09706757  0.29669853], [-0.27048636  0.09719837  0.29500006], [-0.27109119  0.09948142  0.29390262], [-0.27022867  0.09534995  0.29569672], [-0.26894592  0.09508272  0.29735467], [-0.27264398  0.09708
            # [[-0.26795527  0.09939801  0.29517677], [-0.2679654   0.09670212  0.29634369], [-0.26985969  0.09684121  0.29465284], [-0.2704597   0.09912685  0.29355887], [-0.2696089   0.09499108  0.29534846], [-0.26832092  0.09471829  0.2970012 ], [-0.27202182  0.09674
            # [[-0.26776907  0.09929549  0.29495678], [-0.26778029  0.09659567  0.29611912], [-0.26968658  0.09674212  0.29444241], [-0.27029041  0.09903121  0.293358  ], [-0.26943544  0.09488975  0.29513226], [-0.26813579  0.09461097  0.29677437], [-0.27185628  0.09664



            mesh.compute_vertex_normals()
            vis.update_geometry(mesh)
            vis.poll_events()
            vis.update_renderer()


if __name__ == '__main__':

    clock = Clock()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inertial_poser = TransPoseNet(num_past_frame=20, num_future_frame=5).to(device)
    #
    # # normalization
    # # acc = torch.cat((acc_cal[:, :5] - acc_cal[:, 5:], acc_cal[:, 5:]), dim=1).bmm(ori_cal[:, -1]) / config.acc_scale
    # # ori = torch.cat((ori_cal[:, 5:].transpose(2, 3).matmul(ori_cal[:, :5]), ori_cal[:, 5:]), dim=1)
    #
    # acc = torch.load(os.path.join(paths.example_dir, 'acc.pt'))
    # ori = torch.load(os.path.join(paths.example_dir, 'ori.pt'))

    acc = torch.load("data/datassmpl_file et_work/AMASS/vacc.pt")[0]
    ori = torch.load("data/datassmpl_file et_work/AMASS/vrot.pt")[0]
    #
    # data_nn = torch.cat((acc.view(-1, 18), ori.view(-1, 54)), dim=1).to(device)
    # pose, tran = inertial_poser.forward_online(data_nn)
    # pose = rotation_matrix_to_axis_angle(pose.view(1, 216)).view(72)

    length = acc.shape[0]
    A = Main1()

    for j in range(length):
        clock.tick(60)
        acc_ = acc[j]
        ori_ = ori[j]
        data_nn_ = normalize_and_concat(acc_, ori_).to(device)
        # data_nn_ = torch.cat((acc_.view(-1, 18), ori_.view(-1, 54)), dim=1).to(device)
        pose, tran = inertial_poser.forward_online(data_nn_)
        A.view_motion([pose], [tran], j)

        print('\r', 'Sensor FPS:', clock.get_fps(), end='')