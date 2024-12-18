import os

def create_directory_structure():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    datasets = [
        'audio',
        'cifar',
        'deep',
        'enron',
        'gist',
        'glove',
        'imagenet',
        'millionsong',
        'mnist',
        'notre',
        'nuswide',
        'sift',
        'siftsmall',
        'sun',
        'trevi',
        'ukbench',
        'wikipedia-2024-06-bge-m3-zh'
    ]

    test_types = ['panng-test', 'qg-test', 'onng-test', 'qbg-test']

    # Create result directory structure
    for test_type in test_types:
        for dataset in datasets:
            path = os.path.join(project_root, 'result', test_type, f'{dataset}')
            os.makedirs(path, exist_ok=True)

    # Create index directory structure with both regular and refined folders
    for test_type in test_types:
        for dataset in datasets:
            # Regular dataset folder
            path = os.path.join(project_root, 'index', test_type, f'{dataset}')
            os.makedirs(path, exist_ok=True)
            # Refined dataset folder
            path_refined = os.path.join(project_root, 'index', test_type, f'{dataset}_refined')
            os.makedirs(path_refined, exist_ok=True)

    print("Directory structure created successfully!")

if __name__ == '__main__':
    create_directory_structure()