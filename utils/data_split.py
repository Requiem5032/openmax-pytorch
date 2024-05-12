import glob
import sys
import re
import glob
import numpy as np
import pandas as pd


def get_data_map():
    species_map = {}
    genus_dir = glob.glob('data/*/')
    pattern = re.compile(r'([^/]*)(?=/$)')
    for genus in genus_dir:
        genus_name = pattern.search(genus)
        species_dir = glob.glob(genus + '*/')

        for species in species_dir:
            species_name = pattern.search(species)
            gen_sp = ' '.join([genus_name.group(), species_name.group()])
            sub_species_dir = glob.glob(species + '*/')
            species_map[gen_sp] = species

            for sub_species in sub_species_dir:
                sub_species_name = pattern.search(sub_species)
                gen_sp_subsp = ' '.join(
                    [genus_name.group(), species_name.group(), sub_species_name.group()])
                species_map[gen_sp_subsp] = sub_species
    return species_map


def get_species_map(df):
    class_map = {}
    for i in range(len(df)):
        class_map[df.loc[i, 'Species']] = df.loc[i, 'Species_Name']
    return class_map


def rename_species(name):
    pattern = re.compile(r'([^ ]*)(?=$)')
    if pd.isnull(name):
        return name
    assert len(name.split(' ')) > 1
    species = pattern.search(name)
    new_name = name.replace(species.group(), name.replace(' ', '_'))
    return new_name


def row_to_species(split_df, unique_species):
    row_species_map = {}
    for i in range(len(split_df)):
        idx = False
        if not pd.isnull(split_df.loc[i, 'unknown']):
            row_species_map[i] = unique_species.index(
                split_df.loc[i, 'unknown'])
            idx = True
        if not pd.isnull(split_df.loc[i, 'genus']):
            if idx:
                raise ValueError('Resetting directory to species map')
            row_species_map[i] = unique_species.index(split_df.loc[i, 'genus'])
            idx = True
        if not pd.isnull(split_df.loc[i, 'species']):
            if idx:
                raise ValueError('Resetting directory to species map')
            row_species_map[i] = unique_species.index(
                split_df.loc[i, 'species'])
    return row_species_map


def split_helper(id_to_sample, ratio):
    df = pd.DataFrame(data=list(id_to_sample.keys()))
    train_list, val_list, test_list = np.split(
        df.sample(frac=1), [int(ratio[0]*len(df)), int((ratio[0]+ratio[1])*len(df))])
    return train_list, val_list, test_list


def join_list(id_to_sample, id_list):
    joined_list = []
    for id in id_list:
        joined_list.extend(id_to_sample[id])
    return joined_list


def split_img_list(img_list, ratio):
    id_to_sample = {}
    pattern = re.compile(r'(?<=-)\d*(?=_)')
    for img in img_list:
        id_match = pattern.search(img)
        id = id_match.group()
        if id in id_to_sample:
            id_to_sample[id].append(img)
        else:
            id_to_sample[id] = [img]
    train_list, val_list, test_list = split_helper(id_to_sample, ratio)
    try:
        train_list = join_list(id_to_sample, train_list[0])
    except:
        sys.exit(train_list[0])
    val_list = join_list(id_to_sample, val_list[0])
    test_list = join_list(id_to_sample, test_list[0])
    return train_list, val_list, test_list


def split_data(fold_num):
    split_df = pd.read_excel(
        'data/unk training - 1.29.2020 - reduced.xlsx', sheet_name=f'fold{fold_num}')
    split_df = split_df[split_df['ignore'] != 'YES'].reset_index(drop=True)
    split_df['genus'] = split_df['genus'].apply(rename_species)
    split_df['species'] = split_df['species'].apply(rename_species)
    return split_df


def extend_df(output_df, img_list, target_genus, target_species, split, genus_name, species_name):
    temp_df = pd.DataFrame(data=img_list, columns=['Id'])
    temp_df['Genus'] = target_genus
    temp_df['Species'] = target_species
    temp_df['Split'] = split
    temp_df['Genus_Name'] = genus_name
    temp_df['Species_Name'] = species_name
    temp_df = temp_df[output_df.columns]
    output_df = pd.concat([output_df, temp_df]).reset_index(drop=True)
    return output_df


def get_unique_species(fold_num):
    if fold_num == 1:
        unique_species = [
            'aedes aedes_aegypti',
            'aedes aedes_albopictus',
            'aedes aedes_dorsalis',
            'aedes aedes_japonicus',
            'aedes aedes_sollicitans',
            'aedes aedes_vexans',
            'anopheles anopheles_coustani',
            'anopheles anopheles_crucians',
            'anopheles anopheles_freeborni',
            'anopheles anopheles_funestus',
            'anopheles anopheles_gambiae',
            'culex culex_erraticus',
            'culex culex_pipiens_sl',
            'culex culex_salinarius',
            'psorophora psorophora_columbiae',
            'psorophora psorophora_ferox',
            'aedes aedes_spp',
            'anopheles anopheles_spp',
            'culex culex_spp',
            'psorophora psorophora_spp',
            'mosquito']
    elif fold_num == 2:
        unique_species = [
            'aedes aedes_albopictus',
            'aedes aedes_dorsalis',
            'aedes aedes_japonicus',
            'aedes aedes_taeniorhynchus',
            'aedes aedes_vexans',
            'anopheles anopheles_coustani',
            'anopheles anopheles_crucians',
            'anopheles anopheles_funestus',
            'anopheles anopheles_gambiae',
            'anopheles anopheles_punctipennis',
            'anopheles anopheles_quadrimaculatus',
            'culex culex_erraticus',
            'culex culex_salinarius',
            'psorophora psorophora_columbiae',
            'psorophora psorophora_cyanescens',
            'psorophora psorophora_ferox',
            'aedes aedes_spp',
            'anopheles anopheles_spp',
            'culex culex_spp',
            'psorophora psorophora_spp',
            'mosquito']
    elif fold_num == 3:
        unique_species = [
            'aedes aedes_aegypti',
            'aedes aedes_dorsalis',
            'aedes aedes_japonicus',
            'aedes aedes_sollicitans',
            'aedes aedes_taeniorhynchus',
            'anopheles anopheles_coustani',
            'anopheles anopheles_crucians',
            'anopheles anopheles_freeborni',
            'anopheles anopheles_gambiae',
            'anopheles anopheles_punctipennis',
            'anopheles anopheles_quadrimaculatus',
            'culex culex_erraticus',
            'culex culex_pipiens_sl',
            'psorophora psorophora_columbiae',
            'psorophora psorophora_cyanescens',
            'psorophora psorophora_ferox',
            'aedes aedes_spp',
            'anopheles anopheles_spp',
            'culex culex_spp',
            'psorophora psorophora_spp',
            'mosquito']
    elif fold_num == 4:
        unique_species = [
            'aedes aedes_aegypti',
            'aedes aedes_albopictus',
            'aedes aedes_japonicus',
            'aedes aedes_sollicitans',
            'aedes aedes_taeniorhynchus',
            'aedes aedes_vexans',
            'anopheles anopheles_crucians',
            'anopheles anopheles_freeborni',
            'anopheles anopheles_funestus',
            'anopheles anopheles_punctipennis',
            'anopheles anopheles_quadrimaculatus',
            'culex culex_erraticus',
            'culex culex_pipiens_sl',
            'culex culex_salinarius',
            'psorophora psorophora_cyanescens',
            'psorophora psorophora_ferox',
            'aedes aedes_spp',
            'anopheles anopheles_spp',
            'culex culex_spp',
            'psorophora psorophora_spp',
            'mosquito']
    elif fold_num == 5:
        unique_species = [
            'aedes aedes_aegypti',
            'aedes aedes_albopictus',
            'aedes aedes_dorsalis',
            'aedes aedes_sollicitans',
            'aedes aedes_taeniorhynchus',
            'aedes aedes_vexans',
            'anopheles anopheles_coustani',
            'anopheles anopheles_freeborni',
            'anopheles anopheles_funestus',
            'anopheles anopheles_gambiae',
            'anopheles anopheles_punctipennis',
            'anopheles anopheles_quadrimaculatus',
            'culex culex_pipiens_sl',
            'culex culex_salinarius',
            'psorophora psorophora_columbiae',
            'psorophora psorophora_cyanescens',
            'aedes aedes_spp',
            'anopheles anopheles_spp',
            'culex culex_spp',
            'psorophora psorophora_spp',
            'mosquito']
    elif fold_num == 'big':
        unique_species = [
            'aedes aedes_aegypti',
            'aedes aedes_albopictus',
            'aedes aedes_atlanticus',
            'aedes aedes_canadensis',
            'aedes aedes_dorsalis',
            'aedes aedes_flavescens',
            'aedes aedes_infirmatus',
            'aedes aedes_japonicus',
            'aedes aedes_nigromaculis',
            'aedes aedes_sollicitans',
            'aedes aedes_taeniorhynchus',
            'aedes aedes_triseriatus',
            'aedes aedes_trivittatus',
            'aedes aedes_vexans',
            'anopheles anopheles_coustani',
            'anopheles anopheles_crucians',
            'anopheles anopheles_freeborni',
            'anopheles anopheles_funestus',
            'anopheles anopheles_gambiae',
            'anopheles anopheles_pseudopunctipennis',
            'anopheles anopheles_punctipennis',
            'anopheles anopheles_quadrimaculatus',
            'coquillettidia coquillettidia_perturbans',
            'culex culex_coronator',
            'culex culex_erraticus',
            'culex culex_nigripalpus',
            'culex culex_pipiens_sl',
            'culex culex_restuans',
            'culex culex_salinarius',
            'culiseta culiseta_incidens',
            'culiseta culiseta_inornata',
            'deinocerites deinocerites_cancer',
            'deinocerites deinocerites_cuba-1',
            'mansonia mansonia_titillans',
            'psorophora psorophora_ciliata',
            'psorophora psorophora_columbiae',
            'psorophora psorophora_cyanescens',
            'psorophora psorophora_ferox',
            'psorophora psorophora_pygmaea',
            'aedes aedes_spp',
            'anopheles anopheles_spp',
            'culex culex_spp',
            'psorophora psorophora_spp',
            'mosquito']

    if fold_num == 'big':
        unique_genus = [
            'aedes',
            'anopheles',
            'culex',
            'coquillettidia',
            'culiseta',
            'deinocerites',
            'mansonia',
            'psorophora',
            'mosquito'
        ]
    else:
        unique_genus = [
            'aedes',
            'anopheles',
            'culex',
            'psorophora',
            'mosquito'
        ]

    return unique_genus, unique_species


def prepare_data(fold_num, ratio=[0.7, 0.15, 0.15]):
    split_df = split_data(fold_num)
    unique_genus, unique_species = get_unique_species(fold_num)
    species_map = get_data_map()
    row_species_map = row_to_species(split_df, unique_species)

    output_df = pd.DataFrame(
        columns=['Id', 'Genus', 'Species', 'Split', 'Genus_Name', 'Species_Name'])

    for i in range(len(split_df)):
        source_path = species_map[split_df.loc[i, 'folder']]
        img_list = glob.glob(source_path + '*m.jpg')
        target_species = row_species_map[i]
        target_genus = unique_genus.index(
            unique_species[target_species].split(' ')[0])
        genus_name = unique_genus[target_genus]
        if genus_name == 'mosquito':
            species_name = 'mosquito'
        else:
            species_name = unique_species[target_species].split(' ')[1]

        if split_df.loc[i, 'fold'] == 'tr/v/t':
            train_list, val_list, test_list = split_img_list(img_list, ratio)
            output_df = extend_df(output_df, train_list, target_genus,
                                  target_species, 'train', genus_name, species_name)
            output_df = extend_df(output_df, val_list, target_genus,
                                  target_species, 'val', genus_name, species_name)
            output_df = extend_df(output_df, test_list, target_genus,
                                  target_species, 'test', genus_name, species_name)
        else:
            output_df = extend_df(output_df, img_list, target_genus,
                                  target_species, 'ext_test', genus_name, species_name)
    return output_df


def validate_split(split_df, species_map):
    for i in range(len(split_df)):
        unk = split_df.loc[i, 'unknown']
        genus_known = split_df.loc[i, 'genus']
        species_known = split_df.loc[i, 'species']
        assert pd.isnull(unk) + pd.isnull(genus_known) + \
            pd.isnull(species_known) == 2, f'Error at row {i}'
        assert split_df.loc[i, 'folder'] in species_map, 'Error at row {}, {} not found'.format(
            i, split_df.loc[i, 'folder'])
        file_num = len(
            glob.glob(species_map[split_df.loc[i, 'folder']] + '*m.*'))
        if file_num != split_df.loc[i, 'm']:
            print('Index {}, files found in {}: {}, files according to split: {}'.format(
                i, species_map[split_df.loc[i, 'folder']], file_num, split_df.loc[i, 'm']))


# def filter_value(df, col, values, include=False):
#     if include:
#         return df[df[col].isin(values)]
#     else:
#         return df[~df[col].isin(values)]


# def create_dataset(df, batch_size, mode):
#     class_num = len(get_species_map(df))
#     df_known = filter_value(df, 'Species', range(class_num-5, class_num), include=False
#                             ).reset_index(drop=True)
#     class_num_known = len(get_species_map(df_known))

#     if mode == 'train':
#         output_df = filter_value(
#             df_known, 'Split', ['train'], include=True).reset_index(drop=True)
#     elif mode == 'val':
#         output_df = filter_value(
#             df_known, 'Split', ['val'], include=True).reset_index(drop=True)
#     elif mode == 'test':
#         output_df = filter_value(
#             df_known, 'Split', ['test'], include=True).reset_index(drop=True)
#     elif mode == 'test_unknown':
#         output_df = filter_value(
#             df, 'Split', ['test'], include=True).reset_index(drop=True)
#     else:
#         return None

#     output_df = pd.DataFrame(data={
#         'img': output_df['Id'],
#         'label': output_df['Species'],
#     })

#     output_df['img'] = output_df['img'].map(lambda x: process_img_path(x))
#     output_df['label'] = output_df['label'].map(
#         lambda x: encode_one_hot(x, class_num_known))

#     img_list = []
#     label_list = []

#     for i in range(len(output_df)):
#         img_list.append(output_df['img'][i])
#     for i in range(len(output_df)):
#         label_list.append(output_df['label'][i])

#     img_list = np.asarray(img_list, dtype=np.float32)
#     label_list = np.asarray(label_list, dtype=np.float32)

#     output_ds = JHUDataset(x=img_list, y=label_list, batch_size=batch_size)
#     return output_ds
