#include "index.h"

void load_indexes(char * index_path, vector<unordered_map<pair<int, int>, vector<int>, hash_pair>> &indexes, float * centroids_list, int * assignments_list, long int dataset_size, int kmeans_dim, int subspace_num, int kmeans_num_centroid) {
    cout << ">>> Loading index from: " << index_path << endl;

    FILE *ifile_index;
    ifile_index = fopen(index_path,"rb");
    if (ifile_index == NULL) {
        cout << "File " << index_path << "not found!" << endl;
        exit(-1);
    }
    
    // index: centroids + assignments
    int fread_return;
    fread_return = fread(centroids_list, sizeof(float), subspace_num * 2 * kmeans_num_centroid * kmeans_dim, ifile_index);
    fread_return = fread(assignments_list, sizeof(int), subspace_num * 2 * dataset_size, ifile_index);

    for (int i = 0; i < subspace_num; i++) {
        unordered_map<pair<int, int>, vector<int>, hash_pair> index;
        
        for (int j = 0; j < dataset_size; j++) {
            index[pair<int, int>(assignments_list[i * 2 * dataset_size + j], assignments_list[(i * 2 + 1) * dataset_size + j])].push_back(j);
        }

        indexes.push_back(index);
    }

    fclose(ifile_index);
}


void gen_indexes(vector<arma::mat> data_list, vector<unordered_map<pair<int, int>, vector<int>, hash_pair>> &indexes, long int dataset_size, float * centroids_list, int * assignments_list, int kmeans_dim, int subspace_num, int kmeans_num_centroid, int kmeans_num_iters, long int &index_time) {
    struct timeval start_index, end_index;

    for (int subspace_index = 0; subspace_index < subspace_num; subspace_index++) {

        gettimeofday(&start_index, NULL);

        // IMI first half data clustering
        arma::Row<size_t> assignments_first_half;
        arma::mat centroids_first_half;

        mlpack::kmeans::KMeans<> kmeans_first_half(kmeans_num_iters);

        kmeans_first_half.Cluster(data_list[subspace_index * 2], kmeans_num_centroid, assignments_first_half, centroids_first_half);

        // offline index
        for (int i = 0; i < dataset_size; i++) {
            assignments_list[subspace_index * 2 * dataset_size + i] = (float) assignments_first_half(i);
        }
        
        for (int i = 0; i < kmeans_num_centroid; i++) {
            for (int j = 0; j < kmeans_dim; j++) {
                centroids_list[subspace_index * 2 * kmeans_num_centroid * kmeans_dim + i * kmeans_dim + j] = (float) centroids_first_half(j, i);
            }
        }

        // IMI second half data clustering
        arma::Row<size_t> assignments_second_half;
        arma::mat centroids_second_half;

        mlpack::kmeans::KMeans<> kmeans_second_half(kmeans_num_iters);

        kmeans_second_half.Cluster(data_list[subspace_index * 2 + 1], kmeans_num_centroid, assignments_second_half, centroids_second_half);

        // offline index
        for (int i = 0; i < dataset_size; i++) {
            assignments_list[(subspace_index * 2 + 1) * dataset_size + i] = (float) assignments_second_half(i);
        }
        
        for (int i = 0; i < kmeans_num_centroid; i++) {
            for (int j = 0; j < kmeans_dim; j++) {
                centroids_list[(subspace_index * 2 + 1) * kmeans_num_centroid * kmeans_dim + i * kmeans_dim + j] = (float) centroids_second_half(j, i);
            }
        }

        // Generate dictionary to hold online index (which can be generated by loading the offline index and the dataset)
        unordered_map<pair<int, int>, vector<int>, hash_pair> index;
        
        for (int i = 0; i < dataset_size; i++) {
            index[pair<int, int>(assignments_list[subspace_index * 2 * dataset_size + i], assignments_list[(subspace_index * 2 + 1) * dataset_size + i])].push_back(i);
        }

        indexes.push_back(index);

        gettimeofday(&end_index, NULL);
        index_time += (1000000 * (end_index.tv_sec - start_index.tv_sec) + end_index.tv_usec - start_index.tv_usec);

        cout << "Finish indexing the " << subspace_index << "-th subspace. " << endl;
    }
}


void save_indexes(char * index_path, float * centroids_list, int * assignments_list, long int dataset_size, int kmeans_dim, int subspace_num, int kmeans_num_centroid) {
    FILE *ifile_index;
    ifile_index = fopen(index_path,"wb");

    // index: centroids + assignments
    fwrite(centroids_list, sizeof(float), subspace_num * 2 * kmeans_num_centroid * kmeans_dim, ifile_index);

    fwrite(assignments_list, sizeof(int), subspace_num * 2 * dataset_size, ifile_index);

    fclose(ifile_index);

    cout << "Finish saving the index to " << index_path << endl;
}