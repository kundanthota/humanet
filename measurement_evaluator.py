from sklearn.kernel_ridge import KernelRidge 
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import numpy as np
import trimesh
import argparse
from joblib import dump, load

class Human():
    def __init__(self, kernel, alpha, degree, shape_dirs, template, faces):
        self.kernel = kernel
        self.alpha = alpha
        self.degree = degree
        self.shape_model = KernelRidge(alpha = self.alpha, kernel = self.kernel, degree = self.degree)
        self.measures_model = KernelRidge(alpha = self.alpha, kernel = self.kernel, degree = self.degree)
        self.shape_dirs = shape_dirs
        self.template = template
        self.faces = faces

    def fit_measurements(self, X, y):
        return self.measures_model.fit(X, y)

    def fit_shape(self, X, y):
        return self.shape_model.fit(X,y)
    
    def predict_measurements(self, X):
        return self.measures_model.predict(X)
    
    def predict_shape(self, X):
        return self.shape_model.predict(X)
    
    def display_3D(self, X):
        predicted_vertices = self.template + np.dot(self.shape_dirs, np.squeeze(X))
        #trimesh.Trimesh(predicted_vertices, self.faces).show()
        return trimesh.Trimesh(predicted_vertices, self.faces)
    
    def predict_3D_vertices(self, target, actual):
        
        p_vertices = []
        a_vertices = []
        for t,a in zip(target, actual):
            predicted_vertices = self.template + np.dot(self.shape_dirs, np.squeeze(t))
            actual_vertices = self.template + np.dot(self.shape_dirs, np.squeeze(a))
            p_vertices.append(predicted_vertices)
            a_vertices.append(actual_vertices)
        return np.array(p_vertices), np.array(a_vertices)

    def measurement_loss(self, actual, target):
        mae = mean_absolute_error(actual, target)
        std = np.std(abs(actual - target))

        chest_error = mean_absolute_error(actual[:,0], target[:, 0])
        hip_error = mean_absolute_error(actual[:,1], target[:, 1])
        waist_error = mean_absolute_error(actual[:,2], target[:, 2])

        chest_std = np.std(abs(actual[:,0] - target[:, 0]))
        hip_std = np.std(abs(actual[:,1] - target[:, 1]))
        waist_std = np.std(abs(actual[:,2] - target[:, 2]))

        print(f"Measurements Error:")
        print()
        print(f"Over ALL MAE +/- std : {np.round(mae*1000, 2)} +/- {np.round(std*1000, 2)} mm")
        print(f"Chest MAE +/- std : {np.round(chest_error*1000, 2)} +/- {np.round(chest_std*1000, 2)} mm")
        print(f"Hip MAE +/- std : {np.round(hip_error*1000, 2)} +/- {np.round(hip_std*1000, 2)} mm")
        print(f"Waist MAE +/- std : {np.round(waist_error*1000, 2)} +/- {np.round(waist_std*1000, 2)} mm")
    
    def shape_parameters_loss(self, actual, target):
        mae = mean_absolute_error(actual, target)
        std = np.std(abs(actual- target))
        print()
        print(f"Shape Per Parameters Error: \n Mean Absolute Error +/- std : {np.round(mae, 5)} +/- {np.round(std, 5)}")
    
    def per_vertex_shape_loss(self, actual, target):
        pervertex = []
        for each in np.abs(actual-target):
            pervertex.append(np.sum(each)/len(each.flatten()))
        
        mae = np.mean(np.array(pervertex))
        std = np.std(abs(actual- target))
        mape = mean_absolute_percentage_error(actual.flatten(), target.flatten())
        print()
        print(f"3D shape per vertex Error: \n Mean Absolute Error +/- std : {np.round(mae, 5)} +/- {np.round(std, 5)}")

        print(f"Mape : {np.round(mape)}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", type=str, default='polynomial')
    parser.add_argument("--features", type=str, default='ae')
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--gender", type=str, default='male')
    parser.add_argument("--dataset", type=str, default='nomo', help= "nomo or calvis")
    args = parser.parse_args()

    template = np.load(f'data/{args.gender}_template.npy')
    shape_dirs = np.load(f'data/{args.gender}_shapedirs.npy')
    faces =  np.load(f'data/faces.npy')

    human = Human(kernel = args.kernel,\
                  alpha = args.alpha,\
                  degree = args.degree,\
                  template = template,\
                  shape_dirs = shape_dirs,\
                  faces = faces)

    if args.dataset == 'calvis':
        X_train = np.load(f'data/dataloaders/{args.gender}/{args.features}_train_features.npy')
        X_test = np.load(f'data/dataloaders/{args.gender}/{args.features}_test_features.npy')
        X_train_h_w = np.load(f'data/dataloaders/{args.gender}/train_h_w_measures.npy')
        X_test_h_w = np.load(f'data/dataloaders/{args.gender}/test_h_w_measures.npy')

        X_train = np.concatenate([X_train, X_train_h_w], axis = -1)
        X_test = np.concatenate([X_test, X_test_h_w], axis = -1)

        y_measures_train = np.load(f'data/dataloaders/{args.gender}/train_measures.npy')
        y_measures_test = np.load(f'data/dataloaders/{args.gender}/test_measures.npy')

        y_shape_train = np.load(f'data/dataloaders/{args.gender}/train_betas.npy')
        y_shape_test = np.load(f'data/dataloaders/{args.gender}/test_betas.npy')

    else:

        X_train = np.load(f'nomodata/{args.gender}/{args.features}_train_features.npy')
        X_test = np.load(f'nomodata/{args.gender}/{args.features}_test_features.npy')
        X_train_h = np.load(f'nomodata/{args.gender}/train_heights.npy').reshape(len(X_train), 1)
        X_test_h = np.load(f'nomodata/{args.gender}/test_heights.npy').reshape(len(X_test), 1)

        X_train = np.concatenate([X_train, X_train_h], axis = -1)
        X_test = np.concatenate([X_test, X_test_h], axis = -1)

        y_shape_train = np.load(f'nomodata/{args.gender}/train_betas.npy')
        y_shape_test = np.load(f'nomodata/{args.gender}/test_betas.npy')

        y_measures_train = np.load(f'nomodata/{args.gender}/train_measures.npy')
        y_measures_test = np.load(f'nomodata/{args.gender}/test_measures.npy')

    human.fit_measurements(X_train, y_measures_train)
    human.fit_shape(X_train, y_shape_train)

    dump(human,f'weights/{args.dataset}_{args.gender}_krr.pkl')

    del human

    human = load(f'weights/{args.dataset}_{args.gender}_krr.pkl')
    print("Train")
    measurements = human.predict_measurements(X_train)
    shape = human.predict_shape(X_train)
    p_verts, a_verts = human.predict_3D_vertices(shape, y_shape_train)
    
    human.measurement_loss(measurements, y_measures_train)
    human.shape_parameters_loss(shape, y_shape_train)
    human.per_vertex_shape_loss(p_verts, a_verts)
    
    print("Test")
    measurements = human.predict_measurements(X_test)
    shape = human.predict_shape(X_test)
    p_verts, a_verts = human.predict_3D_vertices(shape, y_shape_test)

    human.measurement_loss(measurements, y_measures_test)
    human.shape_parameters_loss(shape, y_shape_test)
    human.per_vertex_shape_loss(p_verts, a_verts)


    
   





