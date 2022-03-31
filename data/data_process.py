import os
import random
import shutil

def split_train_test(data_path):
    battery_room_dirs = os.listdir(data_path)
    for battery_room_dir in battery_room_dirs:
        if battery_room_dir.split('-')[0] == '12':
            battery_dirs = os.listdir(os.path.join(data_path, battery_room_dir))
            random.shuffle(battery_dirs)
            n_sample = len(battery_dirs)
            ratio = int(n_sample*0.8)
            train_battery_dirs = battery_dirs[:ratio]
            test_battery_dirs = battery_dirs[ratio:]
            for train_battery_dir in train_battery_dirs:
                train_battery = os.path.join(data_path, battery_room_dir, train_battery_dir)
                shutil.copyfile(train_battery,
                                os.path.join('.', 'm6_train', train_battery_dir.split('.')[0]+'_' + battery_room_dir.split('-')[1]+'.csv'))

            for test_battery_dir in test_battery_dirs:
                test_battery = os.path.join(data_path, battery_room_dir, test_battery_dir)
                shutil.copyfile(test_battery,
                                os.path.join('.', 'm6_test', test_battery_dir.split('.')[0]+'_' + battery_room_dir.split('-')[1]+'.csv'))


        else:
            battery_dirs = os.listdir(os.path.join(data_path, battery_room_dir))
            random.shuffle(battery_dirs)
            n_sample = len(battery_dirs)
            ratio = int(n_sample * 0.8)
            train_battery_dirs = battery_dirs[:ratio]
            test_battery_dirs = battery_dirs[ratio:]
            for train_battery_dir in train_battery_dirs:
                train_battery = os.path.join(data_path, battery_room_dir, train_battery_dir)
                shutil.copyfile(train_battery, os.path.join('.', 'm6_train', train_battery_dir))

            for test_battery_dir in test_battery_dirs:
                test_battery = os.path.join(data_path, battery_room_dir, test_battery_dir)
                shutil.copyfile(test_battery, os.path.join('.', 'm6_test', test_battery_dir))

if __name__ == '__main__':
    data_path = os.path.join('.', 'M6_v5')
    split_train_test(data_path)


