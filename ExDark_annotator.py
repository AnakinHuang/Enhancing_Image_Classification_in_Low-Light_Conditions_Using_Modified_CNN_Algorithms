import os
import numpy as np
import pandas as pd
import cv2


folder_path = './ExDark_Annno'


def merge_annot(annotations, classes_index):
    annotations['class_index'] = annotations['class'].map(classes_index)
    annotations['area'] = annotations['w'] * annotations['h']
    single_label = annotations.loc[annotations.groupby('image')['area'].idxmax()]

    annotations = (annotations.groupby('image')
                   .agg({'class': lambda x: list(set(x)),
                         'class_index': lambda x: list(set(x)),
                         'fpath': 'first'})
                   .reset_index())

    annotations = pd.merge(annotations, single_label[['image', 'class_index']], on='image', how='left')
    annotations.rename(columns={'class_index_x': 'class_index', 'class_index_y': 'label'}, inplace=True)
    annotations = annotations[['image', 'class', 'class_index', 'label', 'fpath']]

    annotations.to_csv('./image_annotations_merged.csv', index=False)
    return annotations


def retrieve_annot(base_name, image_txt):
    try:
        df = pd.read_csv(base_name + '/' + image_txt, delim_whitespace=True, skiprows=1, header=None, usecols=[0, 1, 2, 3, 4])
        df.columns = ['class', 'x', 'y', 'w', 'h']
        df.insert(0, 'image', image_txt.replace('.txt', ''))
        df['fpath'] = base_name.replace('./ExDark_Annno', './ExDark')
        return df
    except pd.errors.EmptyDataError as e:
        print(f"Warning: {e}: {base_name + '/' + image_txt}")


def process_annot():
    all_annots = []
    for folder in os.listdir(folder_path):
        if os.path.isdir(folder_path + '/' + folder):
            base_name = folder_path + '/' + folder
            for image_txt in os.listdir(base_name):
                if image_txt.endswith('.txt'):
                    file_annots = retrieve_annot(base_name, image_txt)
                    all_annots.append(file_annots)
    return pd.concat(all_annots)


def annot_to_csv():
    df_annots = process_annot()
    df_annots.to_csv('./image_annotations.csv', index=False)
    # df_annots.sample(1000).to_csv('./image_annotations_1000.csv', index=False)


def average_rgb(annotations):
    total_values = {'r': 0, 'g': 0, 'b': 0}
    total_squares = {'r': 0, 'g': 0, 'b': 0}
    total_pixels = {'r': 0, 'g': 0, 'b': 0}
    mean_std = {'r': {}, 'g': {}, 'b': {}}

    for _, row in annotations.iterrows():
        image = cv2.imread(row['fpath'] + '/' + row['image'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Splitting the channels
        r, g, b = cv2.split(image)
        total_values['r'] += np.sum(r)
        total_values['g'] += np.sum(g)
        total_values['b'] += np.sum(b)
        total_squares['r'] += np.sum(np.square(r))
        total_squares['g'] += np.sum(np.square(g))
        total_squares['b'] += np.sum(np.square(b))
        total_pixels['r'] += r.size
        total_pixels['g'] += g.size
        total_pixels['b'] += b.size

    for color in ['r', 'g', 'b']:
        mean = total_values[color] / total_pixels[color] if total_pixels[color] > 0 else 0
        variance = (total_squares[color] / total_pixels[color]) - mean ** 2
        std = np.sqrt(variance) if variance >= 0 else 0
        mean_std[color]['mean'] = mean
        mean_std[color]['std'] = std

    for color in ['r', 'g', 'b']:
        print(f'Mean {color.upper()}:', mean_std[color]['mean'])
        print(f'STD {color.upper()}:', mean_std[color]['std'])
        print(f'Mean {color.upper()} Norm:', mean_std[color]['mean'] / 255)
        print(f'STD {color.upper()} Norm:', mean_std[color]['std'] / 255)

    return mean_std


if __name__ == "__main__":
    annot_to_csv()
