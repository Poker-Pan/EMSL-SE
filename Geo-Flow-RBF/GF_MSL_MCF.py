from scipy import stats
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import math, os, time, copy, sys, random, inspect, psutil, gc, scipy, trimesh, warnings


import pprint as pp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import scipy.io as scio
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
from sklearn.neighbors import KDTree
from tqdm import tqdm
from sklearn.neighbors import KernelDensity
from scipy.spatial import KDTree
warnings.filterwarnings("ignore", message="Can't initialize NVML")



class Utilize(object):
    def __init__(self, Key_Para): 
        super(Utilize, self).__init__()
        self.Key_Para = Key_Para
        self.Dim_time = Key_Para['Dim_time']
        self.Dim_space = Key_Para['Dim_space']

    def make_file(self):
        root = os.getcwd()
        path = root + '/' + self.Key_Para['File_name']
        if not os.path.exists(path):
            os.makedirs(path)
            if self.Key_Para['type_print'] == 'True':
                pass
            elif self.Key_Para['type_print'] == 'False':
                sys.stdout = open(self.Key_Para['File_name'] + '/' + str(self.Key_Para['File_name']) + '-Code-Print.txt', 'w')
            else:
                print('There need code!')
            print('************' + str(self.Key_Para['File_name']) + '************')

    def print_key(self, keyword):
        print('************Key-Word************')
        pp.pprint(keyword)
        print('************************************')

    def setup_seed(self, seed):
        random.seed(seed)

    def mu_0_mu_1(self):
        mu_0, mu_1 = np.array([self.Key_Para['mu_0']]), np.array([self.Key_Para['mu_1']])
        self.Key_Para['mu_0'], self.Key_Para['mu_1'] = mu_0, mu_1

    def sigma_0_sigma_1(self):
        sigma_0 = self.Key_Para['sigma_0']
        sigma_1 = self.Key_Para['sigma_1']

        self.Key_Para['sigma_0'], self.Key_Para['sigma_1'] = sigma_0, sigma_1

    def Time_Space(self):
        Time = np.array(self.Key_Para['Time']).reshape(1,-1)
        
        self.Key_Para['Time'] = Time

    def Gaussian_node_weights(self):
        self.Key_Para['Gaussian_node_weights'] = np.array([[1/3, 1/3, 1/3, 0.2250],
                                [0.0597158717, 0.4701420641, 0.4701420641, 0.1323941527],
                                [0.4701420641, 0.0597158717, 0.4701420641, 0.1323941527],
                                [0.4701420641, 0.4701420641, 0.0597158717, 0.1323941527],
                                [0.7974269853, 0.1012865073, 0.1012865073, 0.1259391806],
                                [0.1012865073, 0.7974269853, 0.1012865073, 0.1259391806],
                                [0.1012865073, 0.1012865073, 0.7974269853, 0.1259391806]]).astype(np.float64)


class Gnerate_node(object):
    def __init__(self, Key_Para):
        super(Gnerate_node, self).__init__()
        self.Key_Para = Key_Para
        self.Dim_time = Key_Para['Dim_time']
        self.Dim_space = Key_Para['Dim_space']
        self.Time = Key_Para['Time']
        self.Num_Nodes_t = Key_Para['Num_Nodes_t']
        self.type_node = Key_Para['type_node']

    def forward(self):
        if self.type_node == 'Load':
            type_surface = self.Key_Para['type_surface']
            if type_surface == 'Dumbbell_combine':
                file_name = type_surface

                load_data = scio.loadmat('./Geo-Flow-RBF/Data_Set/surface_new/' + file_name + '_' + str(self.Key_Para['Num_Nodes_s']) + '.mat')
                nodes_space = load_data['nodes']
                elements = (load_data['triangles'] - 1).astype(int)
                normal = load_data['normal']
                self.Key_Para['nodes_center'] = np.array([0.5, 0.5, 0.5])

            elif type_surface == 'Dumbbell_split':
                file_name = type_surface

                load_data = scio.loadmat('./Geo-Flow-RBF/Data_Set/surface_new/' + file_name + '_' + str(self.Key_Para['Num_Nodes_s']) + '.mat')
                nodes_space = load_data['nodes']
                elements = (load_data['triangles'] - 1).astype(int)
                normal = load_data['normal']
                self.Key_Para['nodes_center'] = np.array([0.5, 0.5, 0.5])
            
            elif type_surface == 'Torus':
                file_name = type_surface

                nodes_space = np.load('./Geo-Flow-RBF/Data_Set/surface_new/' + file_name + '_' + str(self.Key_Para['Num_Nodes_s']) + '.npy')
                elements = np.load('./Geo-Flow-RBF/Data_Set/surface_new/' + file_name + '_' + str(self.Key_Para['Num_Nodes_s']) + '_elements.npy')
                elements = elements
                normal = nodes_space #load_data['normal']
                self.Key_Para['nodes_center'] = np.array([0.5, 0.5, 0.5])

            elif type_surface == 'New_Torus':
                file_name = type_surface

                nodes_space = np.load('./Geo-Flow-RBF/Data_Set/surface_new/' + file_name + '_' + str(self.Key_Para['Num_Nodes_s']) + '.npy')
                elements = np.load('./Geo-Flow-RBF/Data_Set/surface_new/' + file_name + '_' + str(self.Key_Para['Num_Nodes_s']) + '_elements.npy')
                elements = elements
                normal = nodes_space #load_data['normal']
                self.Key_Para['nodes_center'] = np.array([0.5, 0.5, 0.5])


            nodes = nodes_space
            t = np.linspace(self.Time[0, 0], self.Time[0, 1], self.Num_Nodes_t + 1)
            self.Key_Para['dt'] = (self.Time[0, 1] - self.Time[0, 0]) / (self.Num_Nodes_t)
            self.Key_Para['node_normal'] = normal

        elif self.type_node == 'Generate':
            print('There need code!')

        return t, nodes, elements

    def oprater_nodes(self, nodes, elements):
        pt = nodes
        trg = elements

        npt = pt.shape[0]
        ntrg = trg.shape[0]
        normalVec = np.zeros((ntrg, 3))
        trgCenter = np.zeros((ntrg, 3))
        trgArea = np.zeros((ntrg, 1))
        ptArea = np.zeros((npt, 1))
        ptnormal = np.zeros((npt, 3))


        
        for i in range(trg.shape[0]):
            p1, p2, p3 = trg[i, 0], trg[i, 1], trg[i, 2]
            v1, v2, v3 = pt[p1, :], pt[p2, :], pt[p3, :]
            v12 = (v2 - v1).reshape(1, -1)
            v31 = (v1 - v3).reshape(1, -1)
            n = np.cross(v12, -v31, axis=1)
            trgCenter[i, :] = np.mean(np.stack([v1, v2, v3]), axis=0)
            normalVec[i, :] = n / np.linalg.norm(n)
            trgArea[i] = 1 / 2 * np.linalg.norm(n)

            ptArea[p1] = ptArea[p1] + trgArea[i] / 3
            ptArea[p2] = ptArea[p2] + trgArea[i] / 3
            ptArea[p3] = ptArea[p3] + trgArea[i] / 3
            
            # ptnormal[p1, :] = ptnormal[p1, :] + n
            # ptnormal[p2, :] = ptnormal[p2, :] + n
            # ptnormal[p3, :] = ptnormal[p3, :] + n
        
        kdtree = KDTree(nodes)
        for i in range(nodes.shape[0]):
            cur_nodes = nodes[i]
            _, idx = kdtree.query(cur_nodes, k=self.Key_Para['id_node'])
            neighbors = nodes[idx]
            centroid = np.mean(neighbors, axis=0)
            centered_neighbors = neighbors - centroid
            cov_matrix = np.dot(centered_neighbors.T, centered_neighbors) / self.Key_Para['id_node']
            eigvals, eigvecs = np.linalg.eigh(cov_matrix)
            normal = eigvecs[:, 0]
            ptnormal[i] = normal
            

        ptnormal = ptnormal / np.linalg.norm(ptnormal, axis=1).reshape(-1, 1)

        self.Key_Para['tri_normal'] = normalVec
        self.Key_Para['nodes_normal'] = ptnormal
        self.Key_Para['nodes_area'] = ptArea
        self.Key_Para['tri_area'] = trgArea
        self.Key_Para['tri_center'] = trgCenter
        
    def eigenvectors(self, nodes, elements):
        kdtree = KDTree(nodes)
        eigenvector = np.zeros((nodes.shape[0], 3, 3))
        for i in range(nodes.shape[0]):
            cur_nodes = nodes[i]
            _, id_y = kdtree.query(cur_nodes, k=self.Key_Para['id_node'])
            nodes_local = nodes[id_y, :]

            # id_x, _ = np.where(elements==i)
            # sub_elements = elements[id_x, :]
            # id_y = np.unique(sub_elements.reshape(-1), axis=0)
            # nodes_local = nodes[id_y, :]

            c_nodes = np.mean(nodes_local, axis=0)
            P = np.dot((nodes_local - c_nodes).T, (nodes_local - c_nodes))
            eigen_value = np.linalg.eig(P)[0]
            eigen_vector = np.linalg.eig(P)[1]
            idx = np.flip(np.argsort(eigen_value))
            eigen_value = np.flip(np.sort(eigen_value))
            eigen_vector = eigen_vector[:, idx]
            eigenvector[i, :, :] = eigen_vector
        
        self.Key_Para['eigenvector'] = eigenvector

    def Save_vtk(self, nodes, elements, ite=''):
        w = 1
        # points = vtk.vtkPoints()
        # for coord in nodes:
        #     points.InsertNextPoint(coord[0], coord[1], coord[2])

        # polys = vtk.vtkCellArray()
        # for element in elements:
        #     polys.InsertNextCell(len(element))
        #     for node in element:
        #         polys.InsertCellPoint(node)

        # # U = vtk.vtkFloatArray()
        # # U.SetName("U")
        # # U.SetNumberOfComponents(3)
        # # u1 = np.zeros((nodes.shape[0]))
        
        # # for i in range(len(u1)):
        # #     U.InsertNextTuple3(u1[i], 0.0, 0.0)

        # polyData = vtk.vtkPolyData()
        # polyData.SetPoints(points)
        # polyData.SetPolys(polys)

        # writer = vtk.vtkPolyDataWriter()
        # writer.SetFileName(self.Key_Para['File_name'] + '/' + 'Data-surface-' + str(ite) + '.vtk')
        # writer.SetInputData(polyData)
        # writer.Write()


class Solver(object):
    def __init__(self, Key_Para, gen_Nodes):
        super(Solver, self).__init__()
        self.Key_Para = Key_Para
        self.gen_Nodes = gen_Nodes
    
    def nodes_rho_solver(self, elements, cur_dt, dt, temp_nodes, temp_S, id_BDF):
        l0 = scipy.sparse.diags(np.zeros(temp_nodes[0].shape[0]), 0)
        l1 = scipy.sparse.diags(np.ones(temp_nodes[0].shape[0]), 0)
        l2 = self.Key_Para['A_coef_laplce']
        l3 = self.Key_Para['A_coef_grad'][0]
        l4 = self.Key_Para['A_coef_grad'][1]
        l5 = self.Key_Para['A_coef_grad'][2]
        
        if id_BDF == 1:
            a = 1
            BDF_nodes = temp_nodes[0]
            BDF_S = temp_S[0]
        elif id_BDF == 2:
            a = 3/2
            BDF_nodes = 2*temp_nodes[0] - (1/2)*temp_nodes[1]
            BDF_S = 2*temp_S[0] - (1/2)*temp_S[1]
            # self.gen_Nodes.oprater_nodes(temp_nodes[0], elements)
            # rho_0 = 1 / self.Key_Para['nodes_area']
            # S_0 = np.log(rho_0).reshape(-1)
            # self.gen_Nodes.oprater_nodes(temp_nodes[1], elements)
            # rho_1 = 1 / self.Key_Para['nodes_area']
            # S_1 = np.log(rho_1).reshape(-1)
            # a = 3/2
            # BDF_nodes = 2*temp_nodes[0] - (1/2)*temp_nodes[1]
            # BDF_S = 2*((1-self.Key_Para['alpha'])*temp_S[0] + self.Key_Para['alpha']*S_0) - (1/2)*((1-self.Key_Para['alpha'])*temp_S[1] + self.Key_Para['alpha']*S_1)
        elif id_BDF == 3:
            a = 11/6
            BDF_nodes = 3*temp_nodes[0] - (3/2)*temp_nodes[1] + (1/3)*temp_nodes[2]
            BDF_S = 3*temp_S[0] - (3/2)*temp_S[1] + (1/3)*temp_S[2]
        elif id_BDF == 4:
            a = 25/12
            BDF_nodes = 4*temp_nodes[0] - 3*temp_nodes[1] + (4/3)*temp_nodes[2] - (1/4)*temp_nodes[3]
            BDF_S = 4*temp_S[0] - 3*temp_S[1] + (4/3)*temp_S[2] - (1/4)*temp_S[3]


        
        r0 = np.zeros(temp_nodes[0].shape[0])
        r1 = BDF_nodes[:, 0]
        r2 = BDF_nodes[:, 1]
        r3 = BDF_nodes[:, 2]
        r4 = BDF_S
        
        
        Full_A = scipy.sparse.vstack((scipy.sparse.hstack((a*(1/dt)*l1, l0, l0, -l1, l0, l0, self.Key_Para['eta']*l3)), 
                                      scipy.sparse.hstack((l0, a*(1/dt)*l1, l0, l0, -l1, l0, self.Key_Para['eta']*l4)), 
                                      scipy.sparse.hstack((l0, l0, a*(1/dt)*l1, l0, l0, -l1, self.Key_Para['eta']*l5)),
                                      scipy.sparse.hstack((-l2, l0, l0, l1, l0, l0, l0)), 
                                      scipy.sparse.hstack((l0, -l2, l0, l0, l1, l0, l0)), 
                                      scipy.sparse.hstack((l0, l0, -l2, l0, l0, l1, l0)),
                                      scipy.sparse.hstack((l0, l0, l0, l3, l4, l5, a*(1/dt)*l1 - self.Key_Para['eta']*l2)))).tocsr()
        Full_b = np.hstack(((1/dt)*r1, (1/dt)*r2, (1/dt)*r3, r0, r0, r0, (1/dt)*r4))
        new_sovle = scipy.sparse.linalg.spsolve(Full_A, Full_b)
        new_nodes = new_sovle[0:3*temp_nodes[0].shape[0]].reshape(3, -1).T
        new_v = new_sovle[3*temp_nodes[0].shape[0]:6*temp_nodes[0].shape[0]].reshape(3, -1).T
        new_S = new_sovle[6*temp_nodes[0].shape[0]:]

        if id_BDF == 1:
            temp_nodes = [new_nodes, temp_nodes[0]]
            temp_S = [new_S, temp_S[0]]
        elif id_BDF == 2:
            temp_nodes = [new_nodes, temp_nodes[0], temp_nodes[1]]
            temp_S = [new_S, temp_S[0], temp_S[1]]
        elif id_BDF == 3:
            temp_nodes = [new_nodes, temp_nodes[0], temp_nodes[1], temp_nodes[2]]
            temp_S = [new_S, temp_S[0], temp_S[1], temp_S[2]]
        elif id_BDF == 4:
            temp_nodes = [new_nodes, temp_nodes[0], temp_nodes[1], temp_nodes[2], temp_nodes[3]]
            temp_S = [new_S, temp_S[0], temp_S[1], temp_S[2], temp_S[3]]
        return temp_nodes, temp_S, new_v

    def nodes_rho_solver_tangent(self, elements, cur_dt, dt, temp_nodes, temp_S, id_c, id_BDF):
        l0 = scipy.sparse.diags(np.zeros(temp_nodes[0].shape[0]), 0)
        l1 = scipy.sparse.diags(np.ones(temp_nodes[0].shape[0]), 0)
        l2 = self.Key_Para['A_coef_laplce']
        l3 = self.Key_Para['A_coef_grad'][0]
        l4 = self.Key_Para['A_coef_grad'][1]
        l5 = self.Key_Para['A_coef_grad'][2]
        
        if id_BDF == 1:
            a = 1
            BDF_nodes = temp_nodes[0]
            BDF_S = temp_S[0]
        elif id_BDF == 2:
            if id_c == 0:
                self.gen_Nodes.oprater_nodes(temp_nodes[0], elements)
                rho_0 = 1 / self.Key_Para['nodes_area']
                S_0 = np.log(rho_0).reshape(-1)
                self.gen_Nodes.oprater_nodes(temp_nodes[1], elements)
                rho_1 = 1 / self.Key_Para['nodes_area']
                S_1 = np.log(rho_1).reshape(-1)
                a = 3/2
                BDF_nodes = 2*temp_nodes[0] - (1/2)*temp_nodes[1]
                BDF_S = 2*(S_0 + temp_S[0])/2 - (1/2)*(S_1 + temp_S[1])/2
            else:
                a = 3/2
                BDF_nodes = 2*temp_nodes[0] - (1/2)*temp_nodes[1]
                BDF_S = 2*temp_S[0] - (1/2)*temp_S[1]
        elif id_BDF == 3:
            a = 11/6
            BDF_nodes = 3*temp_nodes[0] - (3/2)*temp_nodes[1] + (1/3)*temp_nodes[2]
            BDF_S = 3*temp_S[0] - (3/2)*temp_S[1] + (1/3)*temp_S[2]
        elif id_BDF == 4:
            a = 25/12
            BDF_nodes = 4*temp_nodes[0] - 3*temp_nodes[1] + (4/3)*temp_nodes[2] - (1/4)*temp_nodes[3]
            BDF_S = 4*temp_S[0] - 3*temp_S[1] + (4/3)*temp_S[2] - (1/4)*temp_S[3]


        
        r0 = np.zeros(temp_nodes[0].shape[0])
        r1 = BDF_nodes[:, 0]
        r2 = BDF_nodes[:, 1]
        r3 = BDF_nodes[:, 2]
        r4 = BDF_S
        
        
        Full_A = scipy.sparse.vstack((scipy.sparse.hstack((a*(1/dt)*l1, l0, l0, self.Key_Para['eta']*l3)), 
                                      scipy.sparse.hstack((l0, a*(1/dt)*l1, l0, self.Key_Para['eta']*l4)), 
                                      scipy.sparse.hstack((l0, l0, a*(1/dt)*l1, self.Key_Para['eta']*l5)),
                                      scipy.sparse.hstack((l0, l0, l0, a*(1/dt)*l1 - self.Key_Para['eta']*l2)))).tocsr()
        Full_b = np.hstack(((1/dt)*r1, (1/dt)*r2, (1/dt)*r3, (1/dt)*r4))
        new_sovle = scipy.sparse.linalg.spsolve(Full_A, Full_b)
        new_nodes = new_sovle[0:3*temp_nodes[0].shape[0]].reshape(3, -1).T
        new_S = new_sovle[3*temp_nodes[0].shape[0]:]
        new_v = new_nodes


        if id_BDF == 1:
            temp_nodes = [new_nodes, temp_nodes[0]]
            temp_S = [new_S, temp_S[0]]
        elif id_BDF == 2:
            temp_nodes = [new_nodes, temp_nodes[0], temp_nodes[1]]
            temp_S = [new_S, temp_S[0], temp_S[1]]
        elif id_BDF == 3:
            temp_nodes = [new_nodes, temp_nodes[0], temp_nodes[1], temp_nodes[2]]
            temp_S = [new_S, temp_S[0], temp_S[1], temp_S[2]]
        elif id_BDF == 4:
            temp_nodes = [new_nodes, temp_nodes[0], temp_nodes[1], temp_nodes[2], temp_nodes[3]]
            temp_S = [new_S, temp_S[0], temp_S[1], temp_S[2], temp_S[3]]
        return temp_nodes, temp_S, new_v
      

class Plot_result(object):
    def __init__(self, Key_Para, gen_Nodes):
        super(Plot_result, self).__init__()
        self.Key_Para = Key_Para
        self.gen_Nodes = gen_Nodes
        self.Dim_time = Key_Para['Dim_time']
        self.Dim_space = Key_Para['Dim_space']
        self.Time = Key_Para['Time']

    def plot_rho(self, t, nodes, elements, rho, ite=''):
        if self.Key_Para['Dim_space'] == 3:
            T = t.shape[0]
            fig = plt.figure(figsize=(10, 10))
            plot_nodes = nodes

            surface_rho = (rho[elements[:, 0]] + rho[elements[:, 1]] + rho[elements[:, 2]]) / 3
            if (surface_rho - surface_rho.min()).max() == 0:
                color_rho = surface_rho.reshape(-1, 1)
            else:
                color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)


            ax = fig.add_subplot(1, 1, 1, projection='3d')
            # surf = ax.scatter(plot_nodes[:, 0], plot_nodes[:, 1], plot_nodes[:, 2], c=rho, s=0.5, alpha=0.5)
            surf = ax.plot_trisurf(plot_nodes[:, 0], plot_nodes[:, 1], plot_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
            surf.set_array(color_rho[:, 0])
            ax.set_xlim3d(-1, 1), ax.set_ylim3d(-1, 1), ax.set_zlim3d(-1, 1)
            # ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
            ax.view_init(elev=45, azim=30)
            if self.Key_Para['type_surface'] == 'New_Peanut':
                ax.view_init(elev=5, azim=5)
            elif self.Key_Para['type_surface'] == 'Sphere1':
                ax.view_init(elev=5, azim=-60)
                ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                ax.axis('off')
                ax.grid(None)
                ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            elif self.Key_Para['type_surface'] == 'Dumbbell_combine' or self.Key_Para['type_surface'] == 'Dumbbell_split':
                ax.view_init(elev=5, azim=90)
                ax.set_xlim3d(-1, 1), ax.set_ylim3d(-1, 1), ax.set_zlim3d(-1, 1)
                ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                ax.axis('off')
                ax.grid(None)
                ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            elif self.Key_Para['type_surface'] == 'New_Torus':
                ax.view_init(elev=60, azim=45)
                ax.set_xlim3d(-1, 1), ax.set_ylim3d(-1, 1), ax.set_zlim3d(-1, 1)
                ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                ax.axis('off')
                ax.grid(None)
                ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            
            ax.set_box_aspect([1, 1, 1])
            
            cb = fig.colorbar(surf, ax=[ax])
            cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
            scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
            scale1 = []
            for j in range(0, 5):
                scale1.append('{:.3f}'.format(scale[j]))
            cb.set_ticklabels(scale1)

            plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-rho-' + str(ite) + '.png', bbox_inches='tight', dpi=300)
            plt.close()

    def plot_S(self, t, nodes, elements, S, ite=''):
        if self.Key_Para['Dim_space'] == 3:
            T = t.shape[0]
            fig = plt.figure(figsize=(10, 10))
            plot_nodes = nodes

            surface_rho = (S[elements[:, 0]] + S[elements[:, 1]] + S[elements[:, 2]]) / 3
            if (surface_rho - surface_rho.min()).max() == 0:
                color_rho = surface_rho.reshape(-1, 1)
            else:
                color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)


            ax = fig.add_subplot(1, 1, 1, projection='3d')
            # surf = ax.scatter(plot_nodes[:, 0], plot_nodes[:, 1], plot_nodes[:, 2], c=rho, s=0.5, alpha=0.5)
            surf = ax.plot_trisurf(plot_nodes[:, 0], plot_nodes[:, 1], plot_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
            surf.set_array(color_rho[:, 0])
            ax.set_xlim3d(-1, 1), ax.set_ylim3d(-1, 1), ax.set_zlim3d(-1, 1)
            # ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
            ax.view_init(elev=45, azim=30)
            if self.Key_Para['type_surface'] == 'New_Peanut':
                ax.view_init(elev=5, azim=5)
            elif self.Key_Para['type_surface'] == 'Sphere1':
                ax.view_init(elev=5, azim=-60)
                ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                ax.axis('off')
                ax.grid(None)
                ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            elif self.Key_Para['type_surface'] == 'Dumbbell_combine' or self.Key_Para['type_surface'] == 'Dumbbell_split':
                ax.view_init(elev=5, azim=90)
                ax.set_xlim3d(-1, 1), ax.set_ylim3d(-1, 1), ax.set_zlim3d(-1, 1)
                ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                ax.axis('off')
                ax.grid(None)
                ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            elif self.Key_Para['type_surface'] == 'New_Torus':
                ax.view_init(elev=60, azim=45)
                ax.set_xlim3d(-1, 1), ax.set_ylim3d(-1, 1), ax.set_zlim3d(-1, 1)
                ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                ax.axis('off')
                ax.grid(None)
                ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            
            ax.set_box_aspect([1, 1, 1])
            
            cb = fig.colorbar(surf, ax=[ax])
            cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
            scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
            scale1 = []
            for j in range(0, 5):
                scale1.append('{:.3f}'.format(scale[j]))
            cb.set_ticklabels(scale1)

            plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-S-' + str(ite) + '.png', bbox_inches='tight', dpi=300)
            plt.close()

    def plot_surface(self, t, nodes, elements, ite=''):
        if self.Key_Para['Dim_space'] == 3:
            T = t.shape[0]
            fig = plt.figure(figsize=(10, 10))
            plot_nodes = nodes

            ax = fig.add_subplot(1, 1, 1, projection='3d')
            surf = ax.plot_trisurf(plot_nodes[:, 0], plot_nodes[:, 1], plot_nodes[:, 2], triangles=elements, vmin=0, vmax=1, edgecolor='black', linewidth=0.1, antialiased=False, color='seashell')
            # surf = ax.scatter(plot_nodes[200:210, 0], plot_nodes[200:210, 1], plot_nodes[200:210, 2], c='red', s=2)
            ax.set_xlim3d(-1, 1), ax.set_ylim3d(-1, 1), ax.set_zlim3d(-1, 1)
            # ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
            ax.view_init(elev=45, azim=30)
            # ax.view_init(elev=-45, azim=-135)
            if self.Key_Para['type_surface'] == 'New_Peanut':
                ax.view_init(elev=5, azim=5)
            elif self.Key_Para['type_surface'] == 'Sphere1':
                ax.view_init(elev=5, azim=-60)
                ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                ax.axis('off')
                ax.grid(None)
                ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            elif self.Key_Para['type_surface'] == 'Dumbbell_combine' or self.Key_Para['type_surface'] == 'Dumbbell_split':
                ax.view_init(elev=5, azim=90)
                ax.set_xlim3d(-1, 1), ax.set_ylim3d(-1, 1), ax.set_zlim3d(-1, 1)
                ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                ax.axis('off')
                ax.grid(None)
                ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            elif self.Key_Para['type_surface'] == 'New_Torus':
                ax.view_init(elev=60, azim=45)
                ax.set_xlim3d(-1, 1), ax.set_ylim3d(-1, 1), ax.set_zlim3d(-1, 1)
                ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                ax.axis('off')
                ax.grid(None)
                ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            
            ax.set_box_aspect([1, 1, 1])


            # omega = np.pi/8
            # R = np.array([[np.cos(omega), -np.sin(omega), 0], [np.sin(omega), np.cos(omega), 0], [0, 0, 1]])
            # exact_nodes = np.dot(R, self.Key_Para['nodes_all'][0, :, :].T).T

            # ax = fig.add_subplot(1, 1, 1)
            # ax.scatter(exact_nodes[:50, 0], exact_nodes[:50, 1], c='black', s=3)
            # ax.scatter(plot_nodes[:50, 0], plot_nodes[:50, 1], c='red', s=3)
            # ax.set_xlim(-1, 1), ax.set_ylim(-1, 1)


            plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-surface-' + str(ite) + '.png', bbox_inches='tight', dpi=300)
            plt.close()

    def plot_normal(self, t, nodes, elements, normal, ite=''):
        if self.Key_Para['Dim_space'] == 3:
            T = t.shape[0]
            fig = plt.figure(figsize=(10, 10))
            plot_nodes = nodes

            ax = fig.add_subplot(1, 1, 1, projection='3d')
            surf = ax.quiver(plot_nodes[:, 0], plot_nodes[:, 1], plot_nodes[:, 2], normal[:, 0], normal[:, 1], normal[:, 2], length=0.05, color='black')
            # surf = ax.scatter(plot_nodes[:, 0], plot_nodes[:, 1], plot_nodes[:, 2], c='black', s=0.5)
            ax.set_xlim3d(-1, 1), ax.set_ylim3d(-1, 1), ax.set_zlim3d(-1, 1)
            # ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
            ax.view_init(elev=5, azim=5)
            # ax.view_init(elev=45, azim=30)
            ax.set_box_aspect([1, 1, 1])
            # plt.title('t=%1.3f' %(t))

            plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-normal' + str(ite) + '.png', bbox_inches='tight', dpi=300)
            plt.close()


class Train_loop(object):
    def __init__(self, Key_Para, gen_Nodes, solver, plot_result):
        super(Train_loop, self).__init__()
        self.Key_Para = Key_Para
        self.gen_Nodes = gen_Nodes
        self.solver = solver
        self.plot_result = plot_result

    def train(self):
        def MLS(self, nodes, elements):          
            grad_gamma_matrix = np.zeros((3, nodes.shape[0], nodes.shape[0]))
            lapalce_gamma_matrix = np.zeros((nodes.shape[0], nodes.shape[0]))

            if id_poly == 2:
                kdtree = KDTree(nodes)
                _, all_id_y = kdtree.query(nodes, k=self.Key_Para['id_node'])
                all_nodes_local = nodes[all_id_y, :]
                all_c_nodes = np.mean(all_nodes_local, axis=1)
                all_nodes_minus = all_nodes_local - all_c_nodes[:, np.newaxis, :]
                all_P = np.matmul(all_nodes_minus.transpose(0, 2, 1), all_nodes_minus)
                all_eigen_value, all_eigen_vector = np.linalg.eigh(all_P)
                sorted_indices = np.argsort(all_eigen_value, axis=1)[:, ::-1]
                all_eigen_value_sort = np.take_along_axis(all_eigen_value, sorted_indices, axis=1)
                all_eigenvector_sort = np.take_along_axis(all_eigen_vector, sorted_indices[:, np.newaxis, :], axis=2)

                all_pro_nodes_local = np.matmul((all_nodes_local - nodes[:, np.newaxis, :]), all_eigenvector_sort)
                all_pro_nodes_center = np.matmul((nodes - nodes)[:, np.newaxis, :], all_eigenvector_sort)
                all_pro_nodes_minus = all_pro_nodes_local - all_pro_nodes_center

                # special weight function
                # weight = np.ones(nodes_local.shape[0]) / len(id_y)
                # weight[np.where(np.linalg.norm(nodes_local - cur_nodes, axis=1) == 0)] = 1

                # inverse of squared distance function
                # epsilon = 1e-3
                # weight = 1 / (np.linalg.norm(nodes_local - cur_nodes, axis=1)**2 + epsilon**2)

                # Wendland function
                all_d = np.linalg.norm((all_nodes_local - nodes[:, np.newaxis, :]), axis=2)
                all_D = 1.1 * np.max(all_d, axis=1).reshape(-1, 1)
                all_weight = (1 - all_d/all_D)**4 * (4*all_d/all_D + 1)

                # 
                # d = np.linalg.norm(nodes_local - cur_nodes, axis=1)
                # D = 1.1 * np.max(d)
                # weight = np.exp(-d**2 / (D**2))

                all_b_x_k = np.concatenate([np.ones((all_pro_nodes_minus.shape[0], all_pro_nodes_minus.shape[1]))[:, np.newaxis, :], 
                                            (all_pro_nodes_minus[:, :, 0])[:, np.newaxis, :], (all_pro_nodes_minus[:, :, 1])[:, np.newaxis, :], 
                                            (all_pro_nodes_minus[:, :, 0])[:, np.newaxis, :]**2, (all_pro_nodes_minus[:, :, 0])[:, np.newaxis, :]*(all_pro_nodes_minus[:, :, 1])[:, np.newaxis, :], (all_pro_nodes_minus[:, :, 1])[:, np.newaxis, :]**2], axis=1)
                all_w_k_b_k = all_b_x_k * all_weight[:, np.newaxis, :]
                all_inv_item_left = np.linalg.inv(np.matmul(all_w_k_b_k, all_b_x_k.transpose(0, 2, 1)))

                all_MLS_matrix = np.matmul(all_inv_item_left, all_w_k_b_k)
                all_sub_f1 = all_pro_nodes_local[:, :, 2:]
                all_coef_a = np.matmul(all_MLS_matrix, all_sub_f1).reshape(-1, 6)

                all_z_x = all_coef_a[:, 1]
                all_z_y = all_coef_a[:, 2]
                all_z_xx = 2*all_coef_a[:, 3]
                all_z_xy = all_coef_a[:, 4]
                all_z_yy = 2*all_coef_a[:, 5]

                all_g = 1 + all_z_x**2 + all_z_y**2
                all_g_hat_11 = (1 + all_z_y**2) / all_g
                all_g_hat_12 = -(all_z_x * all_z_y) / all_g
                all_g_hat_21 = -(all_z_x * all_z_y) / all_g
                all_g_hat_22 = (1 + all_z_x**2) / all_g


                all_G1 = np.hstack((np.zeros((all_g.shape[0], 1)), all_g_hat_11.reshape(-1, 1), all_g_hat_12.reshape(-1, 1), np.zeros((all_g.shape[0], 1)), np.zeros((all_g.shape[0], 1)), np.zeros((all_g.shape[0], 1))))
                all_G2 = np.hstack((np.zeros((all_g.shape[0], 1)), all_g_hat_21.reshape(-1, 1), all_g_hat_22.reshape(-1, 1), np.zeros((all_g.shape[0], 1)), np.zeros((all_g.shape[0], 1)), np.zeros((all_g.shape[0], 1))))
                all_temp = np.matmul((all_eigenvector_sort[:, :, 0] + all_coef_a[:, 1:2]*all_eigenvector_sort[:, :, 2])[:, :, np.newaxis], np.matmul(all_G1[:, np.newaxis, :], all_MLS_matrix)) + \
                    np.matmul((all_eigenvector_sort[:, :, 1] + all_coef_a[:, 2:3]*all_eigenvector_sort[:, :, 2])[:, :, np.newaxis], np.matmul(all_G2[:, np.newaxis, :], all_MLS_matrix))

                grad_gamma_matrix[0, np.arange(all_g.shape[0])[:, None], all_id_y] = all_temp[:, 0, :]
                grad_gamma_matrix[1, np.arange(all_g.shape[0])[:, None], all_id_y] = all_temp[:, 1, :]
                grad_gamma_matrix[2, np.arange(all_g.shape[0])[:, None], all_id_y] = all_temp[:, 2, :]

                all_sqrt_g = np.sqrt(all_g)
                all_g_x = 2*all_z_x*all_z_xx + 2*all_z_y*all_z_xy
                all_g_y = 2*all_z_y*all_z_yy + 2*all_z_x*all_z_xy
                all_sqrt_g_x = 1/2 * (1/all_sqrt_g) * all_g_x 
                all_sqrt_g_y = 1/2 * (1/all_sqrt_g) * all_g_y
                all_g_hat_11_x = ((2*all_z_y*all_z_xy * all_g) - ((1 + all_z_y**2) * all_g_x)) / all_g**2
                all_g_hat_12_x = ((-(all_z_xx*all_z_y + all_z_x*all_z_xy) * all_g) - (-(all_z_x * all_z_y) * all_g_x)) / all_g**2
                all_g_hat_21_y = ((-(all_z_yy*all_z_x + all_z_y*all_z_xy) * all_g) - (-(all_z_x * all_z_y) * all_g_y)) / all_g**2
                all_g_hat_22_y = ((2*all_z_x*all_z_xy * all_g) - ((1 + all_z_x**2) * all_g_y)) / all_g**2

                all_A0 = (np.zeros((all_g.shape[0]))).reshape(-1, 1)
                all_A1 = ((1/all_sqrt_g) * (all_sqrt_g_x * all_g_hat_11 + all_sqrt_g_y * all_g_hat_21) + (all_g_hat_11_x + all_g_hat_21_y)).reshape(-1, 1)
                all_A2 = ((1/all_sqrt_g) * (all_sqrt_g_x * all_g_hat_12 + all_sqrt_g_y * all_g_hat_22) + (all_g_hat_12_x + all_g_hat_22_y)).reshape(-1, 1)
                all_A3 = (all_g_hat_11).reshape(-1, 1)
                all_A4 = (all_g_hat_12 + all_g_hat_21).reshape(-1, 1)
                all_A5 = (all_g_hat_22).reshape(-1, 1)

                all_A = np.hstack((all_A0, all_A1, all_A2, 2*all_A3, all_A4, 2*all_A5))

                temp = np.matmul(all_A[:, np.newaxis, :], all_MLS_matrix).reshape(-1, self.Key_Para['id_node'])
                lapalce_gamma_matrix[np.arange(all_g.shape[0])[:, None], all_id_y] = temp
            elif id_poly == 3:
                coef_a = np.zeros((nodes.shape[0], 10))
                kdtree = KDTree(nodes)
                for i in range(nodes.shape[0]):
                    # id_x, _ = np.where(elements==i)
                    # sub_elements = elements[id_x, :]
                    # id_y = np.unique(sub_elements.reshape(-1), axis=0)
                    # nodes_local = nodes[id_y, :]
                    cur_nodes = nodes[i]
                    _, id_y = kdtree.query(cur_nodes, k=self.Key_Para['id_node'])
                    nodes_local = nodes[id_y, :]

                    c_nodes = np.mean(nodes_local, axis=0)
                    P = np.dot((nodes_local - c_nodes).T, (nodes_local - c_nodes))
                    eigen_value, eigen_vector = np.linalg.eig(P)
                    idx = np.flip(np.argsort(eigen_value))
                    eigen_value = np.flip(np.sort(eigen_value))
                    eigenvector_sort = eigen_vector[:, idx]
                    
                    
                    pro_nodes_local = (nodes_local - cur_nodes) @ eigenvector_sort
                    pro_nodes_center = (cur_nodes - cur_nodes).reshape(1, -1) @ eigenvector_sort
                    pro_nodes_minus = pro_nodes_local - pro_nodes_center

                    # special weight function
                    # weight = np.ones(nodes_local.shape[0]) / len(id_y)
                    # weight[np.where(np.linalg.norm(nodes_local - cur_nodes, axis=1) == 0)] = 1

                    # inverse of squared distance function
                    # epsilon = 1e-3
                    # weight = 1 / (np.linalg.norm(nodes_local - cur_nodes, axis=1)**2 + epsilon**2)

                    # Wendland function
                    d = np.linalg.norm(nodes_local - cur_nodes, axis=1)
                    D = 1.1 * np.max(d)
                    weight = (1 - d/D)**4 * (4*d/D + 1)


                    b_x_k = np.vstack((np.ones((1, pro_nodes_minus.shape[0])), (pro_nodes_minus[:, 0]), (pro_nodes_minus[:, 1]), 
                            (pro_nodes_minus[:, 0])**2, (pro_nodes_minus[:, 0])*(pro_nodes_minus[:, 1]), (pro_nodes_minus[:, 1])**2, 
                            (pro_nodes_minus[:, 0])**3, (pro_nodes_minus[:, 0])**2*(pro_nodes_minus[:, 1]), (pro_nodes_minus[:, 0])*(pro_nodes_minus[:, 1])**2, (pro_nodes_minus[:, 1])**3))
                    w_k_b_k = np.dot(b_x_k, np.diag(weight))
                    inv_item_left = np.linalg.inv(w_k_b_k.dot(b_x_k.T))
                    
                    MLS_matrix = inv_item_left.dot(w_k_b_k)
                    sub_f1 = pro_nodes_local[:, 2:]
                    coef_a[i, :] = MLS_matrix.dot(sub_f1).reshape(-1)
                    
                    z_x = coef_a[i, 1]
                    z_y = coef_a[i, 2]
                    z_xx = 2*coef_a[i, 3]
                    z_xy = coef_a[i, 4]
                    z_yy = 2*coef_a[i, 5]
                    
                    g = 1 + z_x**2 + z_y**2
                    g_hat_11 = (1 + z_y**2) / g
                    g_hat_12 = -(z_x * z_y) / g
                    g_hat_21 = -(z_x * z_y) / g
                    g_hat_22 = (1 + z_x**2) / g
                    
                    G1 = np.vstack((0, g_hat_11, g_hat_12, 0, 0, 0, 0, 0, 0, 0)).T
                    G2 = np.vstack((0, g_hat_21, g_hat_22, 0, 0, 0, 0, 0, 0, 0)).T
                    temp = np.dot((eigenvector_sort[:, 0] + coef_a[i, 1:2]*eigenvector_sort[:, 2]).reshape(-1, 1), np.dot(G1, MLS_matrix)) + \
                        np.dot((eigenvector_sort[:, 1] + coef_a[i, 2:3]*eigenvector_sort[:, 2]).reshape(-1, 1), np.dot(G2, MLS_matrix))

                    grad_gamma_matrix[0, i, id_y] = temp[0, :]
                    grad_gamma_matrix[1, i, id_y] = temp[1, :]
                    grad_gamma_matrix[2, i, id_y] = temp[2, :]
                    
                    sqrt_g = np.sqrt(g)
                    g_x = 2*z_x*z_xx + 2*z_y*z_xy
                    g_y = 2*z_y*z_yy + 2*z_x*z_xy
                    sqrt_g_x = 1/2 * (1/sqrt_g) * g_x 
                    sqrt_g_y = 1/2 * (1/sqrt_g) * g_y
                    g_hat_11_x = ((2*z_y*z_xy * g) - ((1 + z_y**2) * g_x)) / g**2
                    g_hat_12_x = ((-(z_xx*z_y + z_x*z_xy) * g) - (-(z_x * z_y) * g_x)) / g**2
                    g_hat_21_y = ((-(z_yy*z_x + z_y*z_xy) * g) - (-(z_x * z_y) * g_y)) / g**2
                    g_hat_22_y = ((2*z_x*z_xy * g) - ((1 + z_x**2) * g_y)) / g**2

                    A0 = 0
                    A1 = (1/sqrt_g) * (sqrt_g_x * g_hat_11 + sqrt_g_y * g_hat_21) + (g_hat_11_x + g_hat_21_y)
                    A2 = (1/sqrt_g) * (sqrt_g_x * g_hat_12 + sqrt_g_y * g_hat_22) + (g_hat_12_x + g_hat_22_y)
                    A3 = g_hat_11
                    A4 = g_hat_12 + g_hat_21
                    A5 = g_hat_22
                    A = np.vstack((A0, A1, A2, 2*A3, A4, 2*A5, 0, 0, 0, 0)).T
                    
                    temp = np.dot(A, MLS_matrix)
                    lapalce_gamma_matrix[i, id_y] = temp[0, :]
            elif id_poly == 4:
                coef_a = np.zeros((nodes.shape[0], 15))
                kdtree = KDTree(nodes)
                for i in range(nodes.shape[0]):
                    # id_x, _ = np.where(elements==i)
                    # sub_elements = elements[id_x, :]
                    # id_y = np.unique(sub_elements.reshape(-1), axis=0)
                    # nodes_local = nodes[id_y, :]
                    cur_nodes = nodes[i]
                    _, id_y = kdtree.query(cur_nodes, k=self.Key_Para['id_node'])
                    nodes_local = nodes[id_y, :]

                    eigenvector = self.Key_Para['eigenvector']
                    pro_nodes_local = (nodes_local - nodes[i, :]) @ eigenvector[i, :, :]
                    pro_nodes_center = (nodes[i, :] - nodes[i, :]).reshape(1, -1) @ eigenvector[i, :, :]
                    pro_nodes_minus = pro_nodes_local - pro_nodes_center

                    # special weight function
                    # weight = np.ones(nodes_local.shape[0]) / len(id_y)
                    # weight[np.where(np.linalg.norm(nodes_local - cur_nodes, axis=1) == 0)] = 1

                    # inverse of squared distance function
                    # epsilon = 1e-3
                    # weight = 1 / (np.linalg.norm(nodes_local - cur_nodes, axis=1)**2 + epsilon**2)

                    # Wendland function
                    d = np.linalg.norm(nodes_local - cur_nodes, axis=1)
                    D = 1.1 * np.max(d)
                    weight = (1 - d/D)**4 * (4*d/D + 1)


                    b_x_k = np.vstack((np.ones((1, pro_nodes_minus.shape[0])), (pro_nodes_minus[:, 0]), (pro_nodes_minus[:, 1]), 
                                        (pro_nodes_minus[:, 0])**2, (pro_nodes_minus[:, 0])*(pro_nodes_minus[:, 1]), (pro_nodes_minus[:, 1])**2, 
                                        (pro_nodes_minus[:, 0])**3, (pro_nodes_minus[:, 0])**2*(pro_nodes_minus[:, 1]), (pro_nodes_minus[:, 0])*(pro_nodes_minus[:, 1])**2, (pro_nodes_minus[:, 1])**3,
                                        (pro_nodes_minus[:, 0])**4, (pro_nodes_minus[:, 0])**3*(pro_nodes_minus[:, 1]), (pro_nodes_minus[:, 0])**2*(pro_nodes_minus[:, 1])**2, (pro_nodes_minus[:, 0])*(pro_nodes_minus[:, 1])**3, (pro_nodes_minus[:, 1])**4))
                    w_k_b_k = np.dot(b_x_k, np.diag(weight))
                    inv_item_left = np.linalg.inv(w_k_b_k.dot(b_x_k.T))
                    
                    MLS_matrix = inv_item_left.dot(w_k_b_k)
                    sub_f1 = pro_nodes_local[:, 2:]
                    coef_a[i, :] = MLS_matrix.dot(sub_f1).reshape(-1)
                    
                    z_x = coef_a[i, 1]
                    z_y = coef_a[i, 2]
                    z_xx = 2*coef_a[i, 3]
                    z_xy = coef_a[i, 4]
                    z_yy = 2*coef_a[i, 5]
                    
                    g = 1 + z_x**2 + z_y**2
                    g_hat_11 = (1 + z_y**2) / g
                    g_hat_12 = -(z_x * z_y) / g
                    g_hat_21 = -(z_x * z_y) / g
                    g_hat_22 = (1 + z_x**2) / g
                    
                    G1 = np.vstack((0, g_hat_11, g_hat_12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).T
                    G2 = np.vstack((0, g_hat_21, g_hat_22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).T
                    temp = np.dot((eigenvector[i, :, 0] + coef_a[i, 1:2]*eigenvector[i, :, 2]).reshape(-1, 1), np.dot(G1, MLS_matrix)) + \
                        np.dot((eigenvector[i, :, 1] + coef_a[i, 2:3]*eigenvector[i, :, 2]).reshape(-1, 1), np.dot(G2, MLS_matrix))
                    
                    grad_gamma_matrix[0, i, id_y] = temp[0, :]
                    grad_gamma_matrix[1, i, id_y] = temp[1, :]
                    grad_gamma_matrix[2, i, id_y] = temp[2, :]
                    
                    sqrt_g = np.sqrt(g)
                    g_x = 2*z_x*z_xx + 2*z_y*z_xy
                    g_y = 2*z_y*z_yy + 2*z_x*z_xy
                    sqrt_g_x = 1/2 * (1/sqrt_g) * g_x 
                    sqrt_g_y = 1/2 * (1/sqrt_g) * g_y
                    g_hat_11_x = ((2*z_y*z_xy * g) - ((1 + z_y**2) * g_x)) / g**2
                    g_hat_12_x = ((-(z_xx*z_y + z_x*z_xy) * g) - (-(z_x * z_y) * g_x)) / g**2
                    g_hat_21_y = ((-(z_yy*z_x + z_y*z_xy) * g) - (-(z_x * z_y) * g_y)) / g**2
                    g_hat_22_y = ((2*z_x*z_xy * g) - ((1 + z_x**2) * g_y)) / g**2

                    A0 = 0
                    A1 = (1/sqrt_g) * (sqrt_g_x * g_hat_11 + sqrt_g_y * g_hat_21) + (g_hat_11_x + g_hat_21_y)
                    A2 = (1/sqrt_g) * (sqrt_g_x * g_hat_12 + sqrt_g_y * g_hat_22) + (g_hat_12_x + g_hat_22_y)
                    A3 = g_hat_11
                    A4 = g_hat_12 + g_hat_21
                    A5 = g_hat_22
                    A = np.vstack((A0, A1, A2, 2*A3, A4, 2*A5, 0, 0, 0, 0, 0, 0, 0, 0, 0)).T
                    
                    temp = np.dot(A, MLS_matrix)
                    lapalce_gamma_matrix[i, id_y] = temp[0, :]


            A_sub_coef_x, A_sub_coef_y, A_sub_coef_z, A_coef_laplce = grad_gamma_matrix[0, :, :], grad_gamma_matrix[1, :, :], grad_gamma_matrix[2, :, :], lapalce_gamma_matrix
            A_coef_grad = [A_sub_coef_x, A_sub_coef_y, A_sub_coef_z]
            self.Key_Para['A_coef_grad'] = A_coef_grad
            self.Key_Para['A_coef_laplce'] = A_coef_laplce


        t, nodes, elements = self.gen_Nodes.forward()
        self.gen_Nodes.oprater_nodes(nodes, elements)
        rho_0 = 1 / self.Key_Para['nodes_area']
        
        nodes_all = np.zeros((1*t.shape[0], nodes.shape[0], nodes.shape[1]))
        v_all = np.zeros((1*t.shape[0], nodes.shape[0], nodes.shape[1]))
        rho_all = np.zeros((1*t.shape[0], nodes.shape[0]))
        S_all = np.zeros((1*t.shape[0], nodes.shape[0]))

        nodes_all[0, :, :] = nodes
        rho_all[0, :] = rho_0.reshape(-1)
        S_all[0, :] = np.log(rho_0.reshape(-1))
        cur_dt = 0
        all_k, all_new_k, all_dt = [], [], []
        k, new_k = 1, 1
        for dt_i in range(t.shape[0]-1):
            print('dt_i', dt_i)
            if dt_i == 0:
                w = 1
                self.plot_result.plot_rho(t, nodes, elements, rho_0, 0)
                self.plot_result.plot_S(t, nodes, elements, S_all[0, :], 0)
                self.plot_result.plot_surface(t, nodes, elements, 0)
                # self.gen_Nodes.Save_vtk(nodes, elements, 0)
            

            self.Key_Para['dt_i'] = dt_i
            dt = t[dt_i+1] - t[dt_i]
            cur_dt = cur_dt + dt / (k**2)
            print('dt: %.15f' %(dt / (k**2)))
            print('cur_dt: %.10f' %cur_dt)
            all_dt.append(dt / (k**2))
            
        
            id_BDF = self.Key_Para['id_BDF']
            if id_BDF == 1:
                if dt_i == 0:
                    temp_nodes = [nodes_all[0, :]]
                    temp_S = [S_all[0, :]]
                    MLS(self, temp_nodes[0], elements)
                    temp_nodes, temp_S, temp_v = self.solver.nodes_rho_solver(elements, cur_dt, dt, temp_nodes, temp_S, id_BDF=1)
                else:
                    MLS(self, temp_nodes[0], elements)
                    temp_nodes, temp_S, temp_v = self.solver.nodes_rho_solver(elements, cur_dt, dt, temp_nodes, temp_S, id_BDF=1)

                rho = np.exp(temp_S[0])
                nodes_all[(dt_i+1), :, :] = temp_nodes[0]
                S_all[(dt_i+1), :] = temp_S[0]
                v_all[(dt_i+1), :, :] = temp_v
                rho_all[(dt_i+1), :] = rho.reshape(-1)
                print('rho_raito: %.10f' %(rho.max()/rho.min()))
            elif id_BDF == 2:
                if dt_i == 0:
                    temp_nodes = [nodes_all[0, :]]
                    temp_S = [S_all[0, :]]
                    MLS(self, temp_nodes[0], elements)
                    temp_nodes, temp_S, temp_v = self.solver.nodes_rho_solver(elements, cur_dt, dt, temp_nodes, temp_S, id_BDF=1)
                else:
                    if dt_i == 1:
                        temp_nodes = [nodes_all[1, :], nodes_all[0, :]]
                        temp_S = [S_all[1, :], S_all[0, :]]
                    
                    # MLS(self, temp_nodes[0], elements)
                    # extrapola_nodes, _, _ = self.solver.nodes_rho_solver(elements, cur_dt, dt, temp_nodes, temp_S, id_BDF=1)
                    extrapola_nodes = [2*temp_nodes[0] - temp_nodes[1]]
                    MLS(self, extrapola_nodes[0], elements)
                    temp_nodes, temp_S, temp_v = self.solver.nodes_rho_solver(elements, cur_dt, dt, temp_nodes, temp_S, id_BDF=2)


                    id_end = temp_S[0].max() / temp_S[0].min()
                    print('id_end:', id_end)
                    id_c = 0
                    while (id_end - 1 > 1e-3) and (dt_i % 100 == 0):
                        # MLS(self, temp_nodes[0], elements)
                        # extrapola_nodes, _, _ = self.solver.nodes_rho_solver_tangent(elements, cur_dt, dt, temp_nodes, temp_S, id_c, id_BDF=1)
                        extrapola_nodes = [2*temp_nodes[0] - temp_nodes[1]]
                        MLS(self, extrapola_nodes[0], elements)
                        temp_nodes, temp_S, temp_v = self.solver.nodes_rho_solver_tangent(elements, cur_dt, dt, temp_nodes, temp_S, id_c, id_BDF=2)
                        
                        id_end = temp_S[0].max() / temp_S[0].min()
                        print('--id_end:', id_end)
                        id_c = id_c + 1
        

                    new_k = math.floor(2 / np.abs(temp_nodes[0][:, 0].max() - temp_nodes[0][:, 0].min()))
                    if new_k >= 3/2:
                        print('new_k:', new_k)
                        k = k * new_k 
                        cur_center = temp_nodes[0].mean(axis=0)
                        temp_nodes = [new_k*(temp_nodes[0] - cur_center), new_k*(temp_nodes[1] - cur_center)]
                        temp_S = [temp_S[0], temp_S[1]]
                    all_k.append(k)
                    all_new_k.append(new_k)
 


                rho = np.exp(temp_S[0])
                nodes_all[(dt_i+1), :, :] = temp_nodes[0] / k
                S_all[(dt_i+1), :] = temp_S[0]
                v_all[(dt_i+1), :, :] = temp_v
                rho_all[(dt_i+1), :] = rho.reshape(-1)
            elif id_BDF == 3:
                if dt_i == 0:
                    temp_nodes = [nodes_all[0, :], 0]
                    temp_S = [S_all[0, :], 0]
                    new_dt = dt**2
                    temp_dt = 0 + new_dt
                    for _ in range(int(dt/new_dt)):
                        MLS(self, temp_nodes[0], elements)
                        temp_nodes, temp_S, temp_v = self.nodes_rho_solver(elements, cur_dt, dt, temp_nodes, temp_S, id_BDF=1)
                        temp_dt = temp_dt + new_dt
                elif dt_i == 1:
                    temp_nodes = [nodes_all[1, :], nodes_all[0, :]]
                    temp_S = [S_all[1, :], S_all[0, :]]
                    # extrapola_nodes = [2*temp_nodes[0] - temp_nodes[1]]
                    MLS(self, temp_nodes[0], elements)
                    extrapola_nodes, _, _ = self.solver.nodes_rho_solver(elements, cur_dt, dt, temp_nodes, temp_S, id_BDF=1)
                    MLS(self, extrapola_nodes[0], elements)
                    temp_nodes, temp_S, temp_v = self.solver.nodes_rho_solver(elements, cur_dt, dt, temp_nodes, temp_S, id_BDF=2) 
                else:
                    if dt_i == 2:
                        temp_nodes = [nodes_all[2, :], nodes_all[1, :], nodes_all[0, :]]
                        temp_S = [S_all[2, :], S_all[1, :], S_all[0, :]]
                    # extrapola_nodes = [3*temp_nodes[0] - 3*temp_nodes[1] + temp_nodes[2]]
                    MLS(self, temp_nodes[0], elements)
                    extrapola_nodes, _, _ = self.solver.nodes_rho_solver(elements, cur_dt, dt, temp_nodes, temp_S, id_BDF=2)
                    MLS(self, extrapola_nodes[0], elements)
                    temp_nodes, temp_S, temp_v = self.solver.nodes_rho_solver(elements, cur_dt, dt, temp_nodes, temp_S, id_BDF=3)

                rho = np.exp(temp_S[0])
                nodes_all[(dt_i+1), :, :] = temp_nodes[0]
                S_all[(dt_i+1), :] = temp_S[0]
                v_all[(dt_i+1), :, :] = temp_v
                rho_all[(dt_i+1), :] = rho.reshape(-1)
                print('rho_raito: %.10f' %(rho.max()/rho.min()))
            elif id_BDF == 4:
                if dt_i == 0:
                    temp_nodes = [nodes_all[0, :]]
                    temp_S = [S_all[0, :]]
                    new_dt = dt**3
                    temp_dt = 0 + new_dt
                    for _ in range(int(dt/new_dt)):
                        MLS(self, temp_nodes[0], elements)
                        temp_nodes, temp_S, temp_v = self.nodes_rho_solver(elements, cur_dt, dt, temp_nodes, temp_S, id_BDF=1)
                        temp_dt = temp_dt + new_dt
                elif dt_i == 1:
                    temp_nodes = [nodes_all[1, :], nodes_all[0, :]]
                    temp_S = [S_all[1, :], S_all[0, :]]
                    # extrapola_nodes = [2*temp_nodes[0] - temp_nodes[1]]
                    MLS(self, temp_nodes[0], elements)
                    extrapola_nodes, _, _ = self.solver.nodes_rho_solver(elements, cur_dt, dt, temp_nodes, temp_S, id_BDF=1)
                    MLS(self, extrapola_nodes[0], elements)
                    temp_nodes, temp_S, temp_v = self.solver.nodes_rho_solver(elements, cur_dt, dt, temp_nodes, temp_S, id_BDF=2) 
                elif dt_i == 2:
                    temp_nodes = [nodes_all[2, :], nodes_all[1, :], nodes_all[0, :]]
                    temp_S = [S_all[2, :], S_all[1, :], S_all[0, :]]
                    # extrapola_nodes = [3*temp_nodes[0] - 3*temp_nodes[1] + temp_nodes[2]]
                    MLS(self, temp_nodes[0], elements)
                    extrapola_nodes, _, _ = self.solver.nodes_rho_solver(elements, cur_dt, dt, temp_nodes, temp_S, id_BDF=2)
                    MLS(self, extrapola_nodes[0], elements)
                    temp_nodes, temp_S, temp_v = self.solver.nodes_rho_solver(elements, cur_dt, dt, temp_nodes, temp_S, id_BDF=3)
                else:
                    if dt_i == 3:
                        temp_nodes = [nodes_all[3, :], nodes_all[2, :], nodes_all[1, :], nodes_all[0, :]]
                        temp_S = [S_all[3, :], S_all[2, :], S_all[1, :], S_all[0, :]]
                    # extrapola_nodes = [4*temp_nodes[0] - 6*temp_nodes[1] + 4*temp_nodes[2] - temp_nodes[3]]
                    MLS(self, temp_nodes[0], elements)
                    extrapola_nodes, _, _ = self.solver.nodes_rho_solver(elements, cur_dt, dt, temp_nodes, temp_S, id_BDF=3)
                    MLS(self, extrapola_nodes[0], elements)
                    temp_nodes, temp_S, temp_v = self.solver.nodes_rho_solver(elements, cur_dt, dt, temp_nodes, temp_S, id_BDF=4)

                rho = np.exp(temp_S[0])
                nodes_all[(dt_i+1), :, :] = temp_nodes[0]
                S_all[(dt_i+1), :] = temp_S[0]
                v_all[(dt_i+1), :, :] = temp_v
                rho_all[(dt_i+1), :] = rho.reshape(-1)
                print('rho_raito: %.10f' %(rho.max()/rho.min()))


            if (self.Key_Para['type_pre_plot'] == 'True' and (dt_i+1) % 100 == 0) or (dt_i == t.shape[0]-2):
                self.plot_result.plot_rho(t, temp_nodes[0], elements, rho, self.Key_Para['dt_i']+1)
                self.plot_result.plot_S(t, temp_nodes[0], elements, temp_S[0], self.Key_Para['dt_i']+1)
                self.plot_result.plot_surface(t, temp_nodes[0], elements, self.Key_Para['dt_i']+1)


            np.save(self.Key_Para['File_name'] + '/' + 'nodes_all.npy', nodes_all)
            np.save(self.Key_Para['File_name'] + '/' + 'rho_all.npy', rho_all)
            np.save(self.Key_Para['File_name'] + '/' + 'S_all.npy', S_all)
            np.save(self.Key_Para['File_name'] + '/' + 'v_all.npy', v_all)
            np.save(self.Key_Para['File_name'] + '/' + 'elements.npy', elements)
            np.save(self.Key_Para['File_name'] + '/' + 'k.npy', np.array(all_k))
            np.save(self.Key_Para['File_name'] + '/' + 'new_k.npy', np.array(all_new_k))
            np.save(self.Key_Para['File_name'] + '/' + 'dt.npy', np.array(all_dt))

            if dt_i == 10005:
                break

        return nodes_all, rho_all



def main(Key_Para):
    utilize = Utilize(Key_Para)
    utilize.make_file()
    utilize.setup_seed(1)
    utilize.Time_Space()
    utilize.print_key(Key_Para)


    gen_Nodes = Gnerate_node(Key_Para)
    solver = Solver(Key_Para, gen_Nodes)
    plot_result = Plot_result(Key_Para, gen_Nodes)
    train_loop = Train_loop(Key_Para, gen_Nodes, solver, plot_result)
    nodes_all, rho_all = train_loop.train()
 



if __name__== "__main__" :
    time_begin = time.time()
    time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
    name = os.path.basename(sys.argv[0])
    File_name = time_now + '-' + name[:-3]

    test = 'Dumbbell_combine'
    if test == 'Dumbbell_combine':
        Dim_time = 1
        Dim_space = 3
        Num_Nodes_t = 50000
        Num_Nodes_s = 1376   # 1376 2296 3558 5652
        Time = [0.0, 50.0]
        eta = 100

        id_poly = 2
        id_BDF = 2
        id_node = 15


        type_print = 'True'           # 'True'      'False' 
        type_surface = 'Dumbbell_combine'  
        type_node = 'Load'         # 'Load'  'Generate' 
        type_pre_plot = 'True'   # 'True'  'False'  


    elif test == 'Dumbbell_split':
        Dim_time = 1
        Dim_space = 3
        Num_Nodes_t = 100000
        Num_Nodes_s = 1272
        Time = [0.0, 10.0]
        eta = 100

        id_poly = 2
        id_BDF = 2
        id_node = 10


        type_print = 'False'           # 'True'      'False' 
        type_surface = 'Dumbbell_split'  
        type_node = 'Load'         # 'Load'  'Generate' 
        type_pre_plot = 'True'   # 'True'  'False'  


    elif test == 'Space-convergence':
        Dim_time = 1
        Dim_space = 3
        Num_Nodes_t = 100
        Num_Nodes_s = 65536    # 256 1024 4096 16384 65536
        Time = [0.0, (1/10000)]
        eta = 100

        id_poly = 2
        id_BDF = 3
        id_node = 10


        type_print = 'False'           # 'True'      'False' 
        type_surface = 'Torus'  
        type_node = 'Load'         # 'Load'  'Generate' 
        type_pre_plot = 'True'   # 'True'  'False'  


    elif test == 'New_Torus':
        Dim_time = 1
        Dim_space = 3
        Num_Nodes_t = 10000
        Num_Nodes_s = 16384    # 4096 16384 65536
        Time = [0.0, 1.0]
        eta = 100

        id_poly = 2
        id_BDF = 2
        id_node = 10


        type_print = 'False'           # 'True'      'False' 
        type_surface = 'New_Torus'  
        type_node = 'Load'         # 'Load'  'Generate' 
        type_pre_plot = 'True'   # 'True'  'False'  








    File_name = File_name + '_' + test + '_BDF' + str(id_BDF) + '_P' + str(id_poly) + '_Nt-' + str(Num_Nodes_t) + '_Ns-' + str(Num_Nodes_s) + '_eta-' + str(eta)
    Key_Parameters = {
        'test': test,
        'File_name': File_name,
        'Dim_time': Dim_time, 
        'Dim_space': Dim_space,
        'Num_Nodes_t': Num_Nodes_t, 
        'Num_Nodes_s': Num_Nodes_s,
        'Time': Time, 
        'eta': eta,
        'id_poly': id_poly,
        'id_BDF':  id_BDF,
        'id_node': id_node,
        'type_print': type_print,
        'type_node': type_node, 
        'type_surface': type_surface,
        'type_pre_plot': type_pre_plot,
            }

    main(Key_Parameters)
    print('Runing_time:', time.time() - time_begin, 's')


