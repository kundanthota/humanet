from body_measurements.measurement import Body3D
import trimesh
import json
import os
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--gender', type = str, required = True,help='male or female')
	parser.add_argument('--path', type = str, required = True, help='path to obj files')
	args = parser.parse_args()
						
	gender_measures = dict()
	measures = dict()
	for subject in sorted(os.listdir(args.path)):
		filename = os.path.join(args.path, subject)
		mesh = trimesh.load(filename)
		try:
			body = Body3D(mesh.vertices, mesh.faces)
			height = body.height()
			weight = body.weight()
		except:
			height = None
			weight = None

		measures[subject] = [height, weight]
	gender_measures[args.gender] = measures

	with open(f'h_w_measures_{args.gender}.json', 'w') as fp:
		json.dump(gender_measures, fp)

		
