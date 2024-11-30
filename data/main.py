import os
import json


def region_rel(c, sudoku, path):
    region_list = sorted(list(sudoku.keys()), key=lambda y: int(y))
    mesh_within_list = []
    region_overlap_list = []
    region_adjoin_list = []
    delta_set = {'Shanghai': 120}
    d = delta_set[c]
    shape = (3, 3)

    for region in region_list:
        region_overlap = list(sudoku[region].keys())
        mesh_within_str = [x + "    within    " + 'region_' + region + '\n' for x in region_overlap]
        mesh_within_list = mesh_within_list + mesh_within_str

        region_overlap.remove(region)
        region_overlap_str = ['region_' + region + "    overlap    " + 'region_' + x + '\n' for x in region_overlap]
        region_overlap_list = region_overlap_list + region_overlap_str

        region_adjoin = [int(region) - shape[0] - d * shape[1], int(region) - shape[0],
                         int(region) - shape[0] + d * shape[1],
                         int(region) - d * shape[1], int(region) + d * shape[1],
                         int(region) + shape[0] - d * shape[1], int(region) + shape[0],
                         int(region) + shape[0] + d * shape[1]]
        region_adjoin = [str(int(x)) for x in region_adjoin]
        region_adjoin = list(set(region_adjoin) & set(region_list))
        region_adjoin_str = ['region_' + region + "    adjoin    " + 'region_' + x + '\n' for x in region_adjoin]
        region_adjoin_list = region_adjoin_list + region_adjoin_str

    with open(path, 'a+') as tf:
        tf.writelines(mesh_within_list)
        tf.writelines(region_overlap_list)
        tf.writelines(region_adjoin_list)


def poi_rel(c, sudoku, path):
    mesh_attributes_path = os.path.join(r'.\CityMeshAttributes', c)
    region_list = sorted(list(sudoku.keys()), key=lambda y: int(y))
    class_list = []
    serve_list = []
    all_mesh_list = []
    for k in region_list:
        region_mesh_list = list(sudoku[k].keys())
        all_mesh_list = all_mesh_list + region_mesh_list
    all_mesh_list = sorted(list(set(all_mesh_list)), key=lambda y: int(y))
    for mesh in all_mesh_list:
        prefix = "poi_" + mesh
        with open(os.path.join(mesh_attributes_path, mesh + '.json')) as jf:
            mesh_dict = json.load(jf)
        mesh_poi = mesh_dict['poi']
        for one_class_poi in mesh_poi:
            name = str(one_class_poi[0])
            num = one_class_poi[1]
            kind = 'class_' + str(one_class_poi[2])
            class_list = class_list + [prefix + "_" + name + "_no." + str(x) +
                                       "    typeof    " + kind + '\n' for x in range(1, num + 1)]
            serve_list = serve_list + [prefix + "_" + name + "_no." + str(x) +
                                       "    serve    " + mesh + '\n' for x in range(1, num + 1)]
    with open(path, 'a+') as tf:
        tf.writelines(class_list)
        tf.writelines(serve_list)


def places_rel(c, sudoku, path):
    mesh_attributes_path = os.path.join(r'.\CityMeshAttributes', c)
    region_list = sorted(list(sudoku.keys()), key=lambda y: int(y))
    level_list = []
    locate_list = []
    all_mesh_list = []
    for k in region_list:
        region_mesh_list = list(sudoku[k].keys())
        all_mesh_list = all_mesh_list + region_mesh_list
    all_mesh_list = sorted(list(set(all_mesh_list)), key=lambda y: int(y))
    for mesh in all_mesh_list:
        prefix = "places_" + mesh
        with open(os.path.join(mesh_attributes_path, mesh + '.json')) as jf:
            mesh_dict = json.load(jf)
        mesh_places = mesh_dict['places']
        for one_class_place in mesh_places:
            name = str(one_class_place[0])
            num = one_class_place[1]
            level = 'level_' + str(one_class_place[2])
            level_list = level_list + [prefix + "_" + name + "_no." + str(x) +
                                       "    districtlevelrank    " + level + '\n' for x in range(1, num + 1)]
            locate_list = locate_list + [prefix + "_" + name + "_no." + str(x) +
                                         "    locateat    " + mesh + '\n' for x in range(1, num + 1)]
    with open(path, 'a+') as tf:
        tf.writelines(level_list)
        tf.writelines(locate_list)


def landuse_rel(c, sudoku, path):
    mesh_attributes_path = os.path.join(r'.\CityMeshAttributes', c)
    region_list = sorted(list(sudoku.keys()), key=lambda y: int(y))
    serve_list = []
    all_mesh_list = []
    for k in region_list:
        region_mesh_list = list(sudoku[k].keys())
        all_mesh_list = all_mesh_list + region_mesh_list
    all_mesh_list = sorted(list(set(all_mesh_list)), key=lambda y: int(y))
    for mesh in all_mesh_list:
        prefix = "landuse_" + mesh
        with open(os.path.join(mesh_attributes_path, mesh + '.json')) as jf:
            mesh_dict = json.load(jf)
        mesh_landuse = mesh_dict['landuse']
        for one_class_landuse in mesh_landuse:
            kind = 'class_' + str(one_class_landuse[2])
            serve_list = serve_list + [mesh + "    function    " + kind + '\n']
    with open(path, 'a+') as tf:
        tf.writelines(serve_list)


def road_connect_rel(c, sudoku, path):
    mesh_attributes_path = os.path.join(r'.\CityMeshAttributes', c)
    region_list = sorted(list(sudoku.keys()), key=lambda y: int(y))
    connect_list = []
    all_mesh_list = []
    for k in region_list:
        region_mesh_list = list(sudoku[k].keys())
        all_mesh_list = all_mesh_list + region_mesh_list
    all_mesh_list = sorted(list(set(all_mesh_list)), key=lambda y: int(y))
    for mesh in all_mesh_list:
        with open(os.path.join(mesh_attributes_path, mesh + '.json')) as jf:
            mesh_dict = json.load(jf)
        mesh_road_connect = mesh_dict['road_connect']
        connect_mesh_list = list(mesh_road_connect.keys())
        connect_list = connect_list + [mesh + "    connect    " + x + '\n' for x in connect_mesh_list]
    with open(path, 'a+') as tf:
        tf.writelines(connect_list)


def sudoku_write(city_set, jsonfile_path, img_fold_path):
    delta_set = {'Shanghai': 120, 'Yantai': 214, 'Hangzhou': 138, 'Jilin': 209, 'Chongqing': 404}

    for sudo_size in [(3, 3), (4, 4)]:
        for city in city_set:
            print("...Writing City " + city)
            sudoku_dict = {}
            save_path = os.path.join(os.getcwd(), str(sudo_size[0]) + "_" + str(sudo_size[1]), city + '.json')
            city_json_path = os.path.join(jsonfile_path, city)
            city_json_list = os.listdir(city_json_path)
            city_json_mesh = [x.replace('.json', '') for x in city_json_list]
            city_img_fold_path = os.path.join(img_fold_path, city)
            img_list = os.listdir(city_img_fold_path)
            img_mesh = [x.replace('.jpg', '') for x in img_list]
            available_mesh_list = list(set(img_mesh) & set(city_json_mesh))

            for centre_mesh_id in available_mesh_list:
                d = delta_set[city]
                a = np.ones(shape=sudo_size) * np.array(range(sudo_size[0]))
                sudoku_id = np.ones(sudo_size[0] * sudo_size[1]) * int(centre_mesh_id) + \
                            (a.T + a * d).reshape(sudo_size[0] * sudo_size[1])

                sudoku_id = sudoku_id.astype(int).tolist()
                sudoku_id = [str(x) for x in sudoku_id]
                sudoku_json = [os.path.join(city_json_path, i + '.json') for i in sudoku_id]

                sudoku_img = [os.path.join(city_img_fold_path, i + '.jpg') for i in sudoku_id]
                path_exists = [os.path.exists(j) for j in sudoku_img]
                if not all(path_exists):
                    continue
                sudoku_dict[centre_mesh_id] = dict.fromkeys(sudoku_id)
                embeddings_list = ['transport_embedding', 'roadnet_embedding',
                                   'function_embedding', 'city_embedding', 'degree_embedding']
                for i, j in enumerate(sudoku_json):
                    with open(j, 'r') as jf:
                        mesh_dict = json.load(jf)
                    sudoku_dict[centre_mesh_id][sudoku_id[i]] = dict([(key, mesh_dict[key]) for key in embeddings_list])

            mesh_json_str = json.dumps(sudoku_dict, indent=4)
            with open(save_path, 'w') as jf:
                jf.write(mesh_json_str)


if __name__ == '__main__':
    city_set = ['Shanghai']
    jsonfile_path = r'.\CityMeshAttributes'
    img_fold_path = r'.\CitySatellite'
    sudoku_write(city_set, jsonfile_path, img_fold_path)
    print('Writing batches successfully!')

    for city in city_set:
        print("......Writing the KG for " + city)
        kg_path = os.path.join(os.getcwd(), 'results', city + '.txt')
        with open(kg_path, 'a+') as f:
            f.truncate(0)

        sudoku_path = os.path.join(r".\CityBatch", city + '.json')
        with open(sudoku_path, 'r') as jf:
            sudoku_dict = json.load(jf)
        sudoku_list = list(sudoku_dict.keys())

        img_path = os.path.join(r".\CitySatellite", city)
        img_list = os.listdir(img_path)
        img_list = [x.replace('.jpg', '') for x in img_list]

        available_top_mesh_list = list(set(sudoku_list) & set(img_list))
        available_sudoku = {k: sudoku_dict[k] for k in available_top_mesh_list}
        available_region_list = list(available_sudoku.keys())
        region_rel(city, available_sudoku, kg_path)
        poi_rel(city, available_sudoku, kg_path)
        places_rel(city, available_sudoku, kg_path)
        landuse_rel(city, available_sudoku, kg_path)
        road_connect_rel(city, available_sudoku, kg_path)
