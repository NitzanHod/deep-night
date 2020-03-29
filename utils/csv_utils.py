import os

from utils.exp_time_extractor import get_exposure_time


def generate_csv(short_folder, long_folder, short_num, csv_path):
    long_imgs = sorted([file for file in os.listdir(long_folder) if file.endswith('.ARW')])
    short_imgs = sorted([file for file in os.listdir(short_folder) if file.endswith('.ARW')])

    short_chunks = [short_imgs[i:i + short_num] for i in range(0, len(short_imgs) - 1, short_num)]

    short_long_match = list(zip(long_imgs, short_chunks))
    with open(csv_path, 'a') as f:
        for long, shorts in short_long_match:
            for short in shorts:
                f.write(os.path.join('./short', short) + ' ' + os.path.join('./long', long) + '\n')


def rename_images_folder(folder):
    imgs = sorted([file for file in os.listdir(folder) if file.endswith('.ARW')])
    ids = [str(int(x.lstrip('DSC').rstrip('.ARW'))) for x in imgs]
    for id in ids:
        rename_path_by_id(folder, id)
    return ids


def rename_path_by_id(root, id):
    if len(str(id)) == 4:
        id = '0' + str(id)
    elif len(str(id)) == 5:
        id = str(id)
    old_path = os.path.join(root, f'DSC{id}.ARW')
    print(old_path)
    time = get_exposure_time(old_path)
    new_path = os.path.join(root, f'{id}_00_{time}s.ARW')
    os.rename(old_path, new_path)


if __name__ == '__main__':
    long_folder = '/home/dsteam/Desktop/mantis_data/sony_260320_yaar-ben-shemen/long'
    short_folder = '/home/dsteam/Desktop/mantis_data/sony_260320_yaar-ben-shemen/short'
    short_num = 2
    csv_path = '../dataset/train_260320_yaar-ben-shemen_list.txt'
    # rename_images_folder(long_folder)
    # rename_images_folder(short_folder)
    generate_csv(short_folder, long_folder, short_num, csv_path)
