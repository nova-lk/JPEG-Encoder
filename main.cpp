#define _USE_MATH_DEFINES

#include <iostream>
#include <cmath>
#include <math.h>
#include <fstream>
#include <random>
#include <iomanip>
#include <map>
#include <sstream>

using namespace std;

struct key_value {
    int key;
    int value;
};

/*Define Functions*/

/*************Session1**************/

void store(float* image, string filename, int size);
float* load(string filename);

/*************Session2**************/

float* generate_uniform_distributed_noise(int size, float lower, float upper);
float* generate_gaussian_distributed_noise(int size, float mean, float std);
float mse(float* image1, float* image2, int size);
float psnr(float* image1, float* image2, int size, int max);
float* add_noise(float* image, float* noise);

/*************Session3**************/

float* generate_dct_basis(int size);
float* transpose(float* inputMatrix, int size);
float* matmul(float* matrix1, float* matrix2, int size);
float* transform(float* image, float* coefficients, int size);
float* threshold(float* dct_coefficients, float threshold);

/*************Session4**************/

float* quantization_table(int q_factor);
float* block_transform(float* image, int image_size, float* dct_basis, int block_size);
float* approximate(float* image, int image_size, float* quant_table, int block_size);
float* quantize(float* image, int image_size, float* quant_table, int block_size);
float* dequantize(float* quantized_image, int image_size, float* quant_table, int block_size);
float* encode(float* image, int image_size, float* quant_table, int block_size);
float* decode(float* quantized_coeff, int image_size, float* quant_table, int block_size);

/*************Session5**************/

float* get_dc_coefficients(float* image, int image_size, int block_size);
void delta_encode(float* coefficients, int size, string filename);
float* delta_decode(string filename);
void run_length_encode(float* input_file, int image_size, int block_size, float* quant_table, string filename);
int* get_zigzag_index(int block_size);
float* run_length_decode(int image_size, int block_size, float* quant_table, string delta_encode, string rle_encode);
map<int, int> calculate_run_length(string filename);
void print_map(std::map<int, int> input);
void print_map(std::map<int, float> input);
key_value get_longest_run(std::map<int, int> input);
std::map<int, float> map_normalize(std::map<int, int> input);
int entropy(std::map<int, float> input);
int total_symbols(std::map<int, int> input);

/*************Session6**************/

string golomb_positive_integers(int num);
string golomb(int num);
int golomb(string code);
string dec_to_binary(int num);
void compress(float* input, string filename);
void decompress(string filename, string image_name);

/*End of functions define*/

int main()
{
    /*************Begining**************/

    /*************Session1**************/

    //to store the pattern
    float* pattern = new float[256 * 256];
    //float* file = new float[256 * 256];

    //create the pattern
    for (int y = 0; y < 256; y++) {
        for (int x = 0; x < 256; x++) {
            pattern[y*256+x] = 0.5f + 0.5f * cos(x * M_PI / 32) * cos(y * M_PI / 64);
        }
    }

    //store pattern
    store(pattern, "pattern.raw", 256);

    //load image
    float* input_file = load("lena_256x256.raw");

    //apply pattern on the image
    float* file_with_pattern = new float[256 * 256];

    for (int y = 0; y < 256; y++) {
        for (int x = 0; x < 256; x++) {
            file_with_pattern[y * 256 + x] = pattern[y * 256 + x] * input_file[y * 256 + x];
        }
    }

    // store the modified image 
    store(file_with_pattern, "file_with_pattern.raw", 256);
    delete[] file_with_pattern;
    delete[] pattern;


    /*************Session2**************/

    // generate an image with uniform-distributed random numbers
    float* uniform_image = generate_uniform_distributed_noise(256, -0.5, 0.5);
    store(uniform_image, "uniform_dist.raw", 256);
    delete[] uniform_image;

    // generate an image with Gaussian-distributed random numbers
    float* gaussian_image = generate_gaussian_distributed_noise(256, 0, 0.288675);
    store(gaussian_image, "gaussian_dist.raw", 256);
    delete[] gaussian_image;


    //load the blurred image
    float* blurred_image = load("lena_blur_256x256.raw");

    // MSE and PSNR of Blur with radius 1
    float psn_ratio = psnr(input_file, blurred_image, 256, 255);
    delete[] blurred_image;

    //add noise to image
    float* gaussian_noise = generate_gaussian_distributed_noise(256, 0, 6.35998);
    float* noise_added_image = add_noise(input_file, gaussian_noise);
    store(noise_added_image, "noisy_lena.raw", 256);
    delete[] noise_added_image;

    // applying blur on noisy image using ImageJ
    //load blurred image and observe MSE
    float* blur_sample = load("nois_blur_lena57.raw");
    psnr(input_file, blur_sample, 256, 255);
    delete[] blur_sample;


    /*************Session3**************/
    
    // create DCT basis vectors
    float* dct_basis256 = generate_dct_basis(256); // returns normalized vectors
    store(dct_basis256, "dct_basis256.raw", 256);

    // Compute the Eucledian distance (l2 norm) of all the basis vectors

    // create non normalized DCT basis vectors
    float* dct_basis_non_norm = new float[256 * 256];
    for (int k = 0; k < 256; k++) {
        for (int n = 0; n < 256; n++) {
            dct_basis_non_norm[k * 256 + n] = cos(M_PI / 256 * (n + 0.5) * k);
        }
    }

    float distance = 0.0f;
    for (int i = 0; i < 256 * 256; i += 256) {
        for (int j = i; j < i + 256; j++) {
            distance += dct_basis_non_norm[j] * dct_basis_non_norm[j];
        }
        std::cout << "Eucledian distance when k = " << i / 256 << " is " << (float)sqrt(distance) << std::endl;
        distance = 0.0f;

    }
    delete[] dct_basis_non_norm;

    // create and save the transpose matrix
    float* dct_basis256T = transpose(dct_basis256, 256);
    store(dct_basis256T, "dct_basis256T.raw", 256);

    // matrix multiplication to check orthonomality
    float* identityMatrix = matmul(dct_basis256, dct_basis256T, 256);
    store(identityMatrix, "identity.raw", 256);

    // produce DCT coefficients from the image
    float* coefficient256 = transform(input_file, dct_basis256, 256);
    store(coefficient256, "transformed.raw", 256);

    // apply different thresholds to DCT coefficients of the image
    float threshold_list[] = {1.0, 5.0, 10.0, 20.0, 50.0, 100.0};

    // iterate through different threshold values to calculate MSE and PSNR
    for (int i = 0; i < sizeof(threshold_list) / sizeof(float); i++) {

        // thresholding
        float* thresholded_image = threshold(coefficient256, threshold_list[i]);
        stringstream stream1; // creating file name to save
        stream1 << "thresholded" << fixed << setprecision(1) << threshold_list[i] << ".raw";
        store(thresholded_image, stream1.str(), 256);

        // reconstruction with IDCT matrix
        float* reconstructed = transform(thresholded_image, dct_basis256T, 256);
        stringstream stream2; // creating file name to save
        stream2 << "reconstructed" << fixed << setprecision(1) << threshold_list[i] << ".raw";
        store(reconstructed, stream2.str(), 256);

        // calculate PSNR
        float psnrReconstructe = psnr(input_file, reconstructed, 256, 255);
        cout << "PSNR of Reconstructed Image (Threshold : " << threshold_list[i] << ") : " << psnrReconstructe << endl;

        delete[] thresholded_image;
        delete[] reconstructed;
    }

    delete[] dct_basis256;
    delete[] dct_basis256T;
    delete[] identityMatrix;
    delete[] coefficient256;
    
    /*************Session4**************/

    // quantization table
    float* quant_table = quantization_table(50);
    store(quant_table, "QTable.raw", 8);

    // create 8x8 DCT basis vectors
    float* dct_basis8x8 = generate_dct_basis(8);
    store(dct_basis8x8, "dct_basis8x8.raw", 8);
    delete[] dct_basis8x8;

    // approximate with 8x8 blocks
    approximate(input_file, 256, quant_table, 8);

    // split approxiamte function to encode and decode
    float* encoded_image = encode(input_file, 256, quant_table, 8);
    float* decoded_image = decode(encoded_image, 256,quant_table, 8);
    store(encoded_image, "encoded.raw", 256);
    store(decoded_image, "decoded.raw", 256);
    delete[] decoded_image;

    /*************Session5**************/

    // 32×32 quantized DC terms
    float* dc = get_dc_coefficients(encoded_image, 256, 8);
    store(dc, "dc_terms.raw", 32);
    delete[] encoded_image;

    // delta encoding of DC terms
    delta_encode(dc, 32, "dc_encode.txt");
    delete[] dc;

    // delta decoding
    float* decoded = delta_decode("dc_encode.txt");
    store(decoded, "decoded_dc_terms.raw", 32);
    delete[] decoded;

    // run length encoding of AC terms
    run_length_encode(input_file, 256, 8, quant_table ,"ac_encode.txt");

    // run length decoding
    float* rle_decoded = run_length_decode(256, 8, quant_table, "dc_encode.txt", "ac_encode.txt");
    store(rle_decoded, "rle_decoded.raw", 256);
    delete[] rle_decoded;
    delete[] quant_table;

    // check value and run length of quantized coefficients
    std::map<int, int> symbol_map = calculate_run_length("ac_encode.txt");
    print_map(symbol_map);

    // find the longest run length
    key_value longest_run = get_longest_run(symbol_map);
    std::cout << "Longest run length: " << longest_run.key << " -> " << longest_run.value << std::endl;

    // runs necessary to encode all AC coefficients
    std::cout << "Total runs needed to encode all AC coefficients: " << symbol_map.size() << std::endl;

    // normalize run length map
    std::map<int, float> norm_symbol_map = map_normalize(symbol_map);
    print_map(norm_symbol_map);

    // theoretical minimum number of bits per symbol
    int minimum_bits = entropy(norm_symbol_map);
    std::cout << "Minimum bits/symbol: " << minimum_bits << std::endl;

    float min_filesize = (float) minimum_bits * total_symbols(symbol_map) / (8 * 1024);
    std::cout << "Minimum possible file size to store AC coefficients with RLE: " << min_filesize << " kB" << std::endl;


    /*************Session6**************/

    // Exponential-Golomb variable-length code
    string code = golomb(1);

    // inverse golomb
    int int_value = golomb(code);

    // test golab and inverse golab
    int int_list[] = {0, -1, 1, -2, 2, -3, 3};
    string golomb_values[7];

    // print result
    std::cout << std::endl;
    for (int i : int_list) {
        std::cout << i << " -> " << golomb(i) <<  " -> "  << golomb(golomb(i)) << std::endl;
    }

    // read an image and generate a compressed bit stream
    compress(input_file, "compressed.txt");
    delete[] input_file;

    // read bits from a text file and reconstruct the image
    decompress("compressed.txt", "decompressed.raw");

    /*************End**************/
}

/**
* @brief Reconstruct an image using a given compressed data file and save it with a given file name.
* @param filename Compressed data file
* @param image_name Name of the output image file
*/
void decompress(string filename, string image_name)
{
    ifstream ifs(filename); // open compressed file

    if (ifs) {

        ofstream delta("temp_dc.txt"); // to store dc coefficients
        ofstream rle("temp_ac.txt"); // to store ac coefficients

        char input; // current reading bit from the input stream
        string code = ""; // golomb code
        bool search = 1; // whether we are searching for a golomb code or decoding one
        int bit_count = 0; // counter to detect one golomb code
        int count = 0; // count of golomb codes -> to separate DC & AC coefficients

        while (ifs >> input) { // read bit by bit
            if (input == '0' && search) { // if input is zero, and we are searching
                bit_count++; // increase no. of bits passed untill find the begining of a code
            }
            else {
                bit_count--; 
                search = 0; 
            }

            code += input; // add read bits into golomb code

            if (bit_count == -1) { // then, we are at end of the golomb code
                if (count < 1024) { // take first 1024 decoded codes as DC coefficients
                    delta << golomb(code) << std::endl;
                    count++;
                }
                else { // rest of the codes are AC coefficients
                    rle << golomb(code) << std::endl;
                }
                bit_count = 0; // reset bit count
                code = ""; // reset golomb code
                search = 1; // change mode back to search
            }
        }

        delta.close();
        rle.close();

        float* quant_table = quantization_table(50); // quantization table for q factor = 50
        float* reconstructed_image = run_length_decode(256, 8, quant_table, "temp_dc.txt", "temp_ac.txt");
        delete[] quant_table;

        // remove temporary files
        remove("temp_dc.txt");
        remove("temp_ac.txt");

        // store reconstructed image
        store(reconstructed_image, image_name, 256);
        delete[] reconstructed_image;
    }
    else {
        std::cout << "Can't open " << filename << std::endl;
    }

    ifs.close();
}

/**
* @brief Compress a given image
* @param input Image file
* @param filename Name of the output file
*/
void compress(float* input, string filename)
{
    float* quant_table = quantization_table(50); // quantization table for q factor = 50
    float* encoded = encode(input, 256, quant_table, 8); // quantized coefficients
    float* dc = get_dc_coefficients(encoded, 256, 8); // dc terms
    delete[] encoded;

    // delta encode dc terms
    delta_encode(dc, 32, "dc_enc.txt"); 
    delete[] dc;

    // run length encode ac terms
    run_length_encode(input, 256, 8, quant_table, "ac_enc.txt");

    delete[] quant_table;
    
    ofstream bitstream(filename); // output bitstream
    ifstream delta("dc_enc.txt"); // read delta encoded DC values

    if (delta)
    {
        string line;
        int value;

        while (getline(delta, line)) {
            value = std::stoi(line);
            bitstream << golomb(value);
        }
        delta.close();
    }

    ifstream rle("ac_enc.txt"); // read run length encoded AC values

    if (rle)
    {
        string line;
        int value;

        while (getline(rle, line)) {
            value = std::stoi(line);
            bitstream << golomb(value);
        }
        rle.close();
    }

    bitstream.close();
}

/**
* @brief Find the integer value of a Golomb code
* @param code Golomb code
* @return Integer value of the Golomb code
*/
int golomb(string code)
{
    // ref: https://stackoverflow.com/questions/23596988/binary-string-to-integer-with-atoi
    // ref: https://en.wikipedia.org/wiki/Exponential-Golomb_coding

    int value = std::stoi(code, nullptr, 2);

    bool odd = value % 2 == 1;

    if (odd)
        return (value - 1) / -2;
    else
        return value / 2;
}

/**
* @brief Find the Golomb code of an integer
* @param num Positive integer
* @return Golomb code
*/
string golomb(int num)
{
    // ref: https://en.wikipedia.org/wiki/Exponential-Golomb_coding

    if (num > 0) {
        // map positive integers to 2x-1
        num = 2 * num - 1;
    }
    else {
        // map negative integers -2x
        num = -2 * num;
    }

    string binary = dec_to_binary(num + 1);

    int digits = binary.size();

    string zeros = "";

    while (digits - 1 > 0) {
        zeros += "0";
        digits--;
    }

    string code = zeros + binary;

    return code;
}

/**
* @brief Find the Golomb code of a positive integer
* @param num Positive integer
* @return Golomb code
*/
string golomb_positive_integers(int num)
{
    // ref: https://en.wikipedia.org/wiki/Exponential-Golomb_coding

    string binary = dec_to_binary(num + 1);

    int digits = binary.size();

    string zeros = "";

    while (digits - 1 > 0) {
        zeros += "0";
        digits--;
    }

    string code = zeros + binary;

    return code;
}

/**
* @brief Convert a decimal value to a binary stream
* @param num Integer in base 10
* @return Binary value
*/
string dec_to_binary(int num)
{
    // ref: https://www.geeksforgeeks.org/reverse-a-string-in-c-cpp-different-methods/

    string binary = "";

    while (num > 0) {
        binary += std::to_string(num % 2);
        num = num / 2;
    }

    // reverse the string
    int n = binary.size();
    for (int i = 0; i < n / 2; i++)
        swap(binary[i], binary[n - i - 1]);

    return binary;
}

/**
* @brief Find the total number of symbols in a given map
* @param input Map
* @return Total number of symbols
*/
int total_symbols(std::map<int, int> input)
{
    int total = 0;

    // calculate the total sybmol count
    for (auto it = input.cbegin(); it != input.cend(); ++it) {
        total += it->second;
    }

    return total;
}

/**
* @brief Find the theoretical minimum number of bits per symbol of a given normalized map
* @param input Normalized map
* @return The minimum number of bits per symbol
*/
int entropy(std::map<int, float> input)
{
    //ref: https://machinelearningmastery.com/what-is-information-entropy/

    float entropy = 0.0f;

    for (auto it = input.cbegin(); it != input.cend(); ++it) {
        entropy -= it->second * log2(it->second);
    }

    return ceil(entropy);
}

/**
* @brief Normalize the values of a given map
* @param input Map to be normalized
* @return Normalized map
*/
std::map<int, float> map_normalize(std::map<int, int> input)
{
    int total = total_symbols(input);

    // normalized map
    std::map<int, float> normalized_map;

    for (auto it = input.cbegin(); it != input.cend(); ++it) {
        normalized_map.insert(std::pair<int, float>(it->first, (float)it->second / total));
    }

    return normalized_map;
}

/**
* @brief Find the symbol with the longest run length from a symbol & run length map
* @param input Map
* @return Symbol and run length of the symbol with the longest run length
*/
key_value get_longest_run(std::map<int, int> input)
{
    // ref: https://stackoverflow.com/questions/9370945/finding-the-max-value-in-a-map

    int max_value = 0;
    int max_key = 0;

    for (auto it = input.cbegin(); it != input.cend(); ++it) {
        if (it->second > max_value) {
            max_key = it->first;
            max_value = it->second;
        }
    }

    key_value longest_run;
    longest_run.key = max_key;
    longest_run.value = max_value;

    return longest_run;
}

/**
* @brief Print keys & values of a map
* @param input Map to be printed
*/
void print_map(std::map<int, int> input)
{
    //ref: https://stackoverflow.com/questions/14070940/how-can-i-print-out-c-map-values
    
    for (auto it = input.cbegin(); it != input.cend(); ++it)
    {
        std::cout << "{" << it->first << ": " << it->second << "}" << std::endl;
    }
    
}

/**
* @brief Print keys & values of a map
* @param input Map to be printed
*/
void print_map(std::map<int, float> input)
{
    //ref: https://stackoverflow.com/questions/14070940/how-can-i-print-out-c-map-values

    for (auto it = input.cbegin(); it != input.cend(); ++it)
    {
        std::cout << "{" << it->first << ": " << it->second << "}" << std::endl;
    }

}

/**
* @brief Calculate the run lengths of different symbols
* @param filename Input file name
* @return Map containing symbols and run lengths
*/
std::map<int, int> calculate_run_length(string filename)
{
    // ref: https://stackoverflow.com/questions/15151480/simple-dictionary-in-c
    // ref: https://stackoverflow.com/questions/36428810/increment-the-value-of-a-map

    std::map<int, int> symbol_map;

    ifstream ifs(filename);
  
    if (ifs) {

        string line;
        int key;
        
        while (getline(ifs, line)) {
            key = std::stoi(line);

            if (symbol_map.find(key) == symbol_map.end()) {
                symbol_map.insert(std::pair<int, int>(key, 1));
            }
            else {
                symbol_map[key]++;
            }
        }
    }

    ifs.close();

    return symbol_map;

}

/**
* @brief Decode a file encoded with run length encoding (RLE)
* @param image_size Image size
* @param block_size Block size
* @param quant_table Quantization weights
* @param delta_encode File name of the delta encoded coefficients
* @param rle_encode File name of the RLE encoded coefficients
* @return Reconstructed image
*/
float* run_length_decode(int image_size, int block_size, float* quant_table, string delta_encode, string rle_encode)
{
    // get the DC coefficinets
    float* delta_decoded = delta_decode(delta_encode);

    float* rle_decoded = new float[image_size * image_size];

    const int EOB = 59; // end of block

    int rle_index = 0; // index of RLE
    int delta_index = 0; // index of Delta

    // first coefficient is the DC coefficient
    rle_decoded[rle_index++] = delta_decoded[delta_index++];

    ifstream ifs(rle_encode);

    // read run-length encoded AC coefficients
    if (ifs) {

        string line;

        // find the length of the file
        while (getline(ifs, line)) {

            int val = std::stoi(line);

            if (val != EOB) {
                rle_decoded[rle_index++] = val;
            }
            else { // if EOB, fill the rest with zeros
                while (rle_index % 64 != 0) {
                    rle_decoded[rle_index++] = 0;
                }
                // fill the DC coefficient of the next block
                rle_decoded[rle_index++] = delta_decoded[delta_index++];
            }

        }
        ifs.close();
    }
    else {
        std::cout << "RLE Enoded file is not found!" << std::endl;
    }

    float* dct_basis = generate_dct_basis(block_size);
    float* idct_basis = transpose(dct_basis, block_size);

    // get the indices in zigzag pattern for the block
    int* zigzag_index = get_zigzag_index(block_size);

    rle_index = 0;

    float* reconstructed_image = new float[image_size * image_size];
    int dc_coeff = 0;

    for (int x = 0; x < image_size; x += block_size) {
        for (int y = 0; y < image_size; y += block_size) {

            // split image into small blocks
            float* block = new float[block_size * block_size];

            // fill the block using zigzag pattern decoding
            for (int i = 0; i < 64; i++) {
                block[zigzag_index[i]] = rle_decoded[rle_index++];
            }
           
            // DC coefficients
            //dc_coeff += block[0];
            //block[0] = dc_coeff;

            // dequantize
            float* dequantized_block = new float[block_size * block_size];

            for (int a = 0; a < block_size; a++) {
                for (int b = 0; b < block_size; b++) {
                    dequantized_block[a * block_size + b] = block[a * block_size + b] * quant_table[a * block_size + b];
                }
            }

            delete[] block;

            // IDCT
            float* reconstructed_block = transform(dequantized_block, idct_basis, block_size);
            delete[] dequantized_block;

            // reassemble the original image
            for (int a = 0; a < block_size; a++) {
                for (int b = 0; b < block_size; b++) {
                    reconstructed_image[(x + a) * image_size + (y + b)] = reconstructed_block[a * block_size + b];
                }
            }
            delete[] reconstructed_block;
        }
    }

    delete[] dct_basis;
    delete[] idct_basis;
    delete[] delta_decoded;
    delete[] rle_decoded;
    delete[] zigzag_index;

    return reconstructed_image;

}

/**
* @brief Encode the AC coefficients of a image using Run Length Encoding (RLE)
* @param input_file Input image file
* @param image_size Size of the input image file
* @param block_size Block size to perform quantization
* @param quant_table Quantization weights matrix
* @param filename Name of the output file that store AC coefficients
*/
void run_length_encode(float* input_file, int image_size, int block_size, float* quant_table, string filename)
{
    // quantized coefficients
    float* encoded_image = encode(input_file, 256, quant_table, 8);

    ofstream ofs(filename);

    const int THREASHOLD = 10; // threshold for RLE
    const int EOB = 59; // sybmbol for end of block

    for (int x = 0; x < image_size; x += block_size) {
        for (int y = 0; y < image_size; y += block_size) {

            // split image into small blocks
            float* quantized_block = new float[block_size * block_size];

            for (int a = 0; a < block_size; a++) {
                for (int b = 0; b < block_size; b++) {
                    quantized_block[a * block_size + b] = encoded_image[(x + a) * image_size + (y + b)];
                }
            }

            // get the indices in zigzag pattern for the block
            int* zigzag_index = get_zigzag_index(block_size);

            int val1 = quantized_block[zigzag_index[1]]; //zigzag_index[0] is the DC coefficient
            int count = 1;

            

            for (int index = 2; index < block_size * block_size; index++) {

                int val2 = quantized_block[zigzag_index[index]];
                
                if (val2 == val1) { // if two adjacents are equal, increase the counter
                    count++;
                }
                else {
                    /*
                    THREASHOLD = 10 prevents text file being writted as multiple columns.
                    * This has done for the simplicity of the implementation.
                    * Also, it has been noticed that other than ending 0, there aren't many recurring symbols.
                    */
                    if (count > THREASHOLD) { // write value and run to the file
                        ofs << std::to_string(val1) << " " << std::to_string(count) << endl;
                    }
                    else { // write only the value to the file
                        
                        while (count > 0) {
                            ofs << std::to_string(val1) << endl;
                            count--;
                        }
                    }
                    count = 1; // reset the counter

                }
                val1 = val2;
            }
            // to mark the end-of-block
            ofs << std::to_string(EOB) << endl;

            delete[] quantized_block;
        }
    }
    ofs.close();

}

/**
* @brief Get the zig-zag index of an N*N matrix
* @param block_size Size of the matrix
* @return Array with indexes of zig-zag traverse pattern
*/
int* get_zigzag_index(int block_size)
{
    int a = 0; // x coordinate
    int b = 1; // y coordinate
    bool go_up = 0; // whether to go up or down in the zigzag pattern

    int* indices = new int[block_size * block_size];

    // first first element
    indices[0] = 0;

    if (block_size > 1) {
        // add second element
        indices[1] = 1;
        int filled = 2;

        while (filled < block_size * block_size) {
            if (go_up) {
                a -= 1;
                b += 1;
                indices[filled++] = a * block_size + b;
                if (a == 0 || b == (block_size - 1)) { // change the direction at the boundary

                    if (a == 0 && b == (block_size - 1)) {
                        a += 1;
                    }
                    else if (b == block_size - 1) {
                        a += 1;
                    }
                    else {
                        b += 1;
                    }

                    indices[filled++] = a * block_size + b;
                    go_up = 0; // go down
                }
            }
            else {
                a += 1;
                b -= 1;
                indices[filled++] = a * block_size + b;
                if (a == (block_size - 1) || b == 0) { // change the direction at the boundary

                    if (a == (block_size - 1) && b == 0) {
                        b += 1;
                    }
                    else if (b == 0) {
                        a += 1;
                    }
                    else {
                        b += 1;
                    }

                    indices[filled++] = a * block_size + b;
                    go_up = 1; // go up

                }
            }
        }
    }

    return indices;
}

/**
* @brief Decode a file encoded with Delta encoding
* @param filename Name of the input file
* @return DC coefficients
*/
float* delta_decode(string filename)
{
    ifstream ifs(filename);

    if (ifs)
    {
        float old = 0;
        float current = 0;
        int length = 0;
        string line;

        
        // find the length of the file
        while (getline(ifs, line)) {
            ++length;
        }
        ifs.close(); // cant use ifs further, it is empty now
        
        // open the same file again to read lines
        ifstream ifs2(filename);
        float* decode = new float[length];
        
        // fill the DC coefficients
        for (int i = 0; i < length; i++) {
            ifs2 >> current;
            decode[i] = current + old;
            old = decode[i];
        }

        ifs2.close();

        return decode;
    }
    else {
        std::cout << "Can't open file " << filename << std::endl;
        return nullptr;
    }

    

}

/**
* @brief Delta encode DC coefficients of an image
* @param coefficients DC coefficients
* @param size Size of the DC coefficients matrix
* @param filename Name of the output file
*/
void delta_encode(float* coefficients, int size, string filename)
{
    ofstream ofs(filename);

    ofs << coefficients[0] << endl;

    for (int i = 1; i < size * size; i++) {
        ofs << coefficients[i] - coefficients[i-1] << endl;
    }

    ofs.close();
}

/**
* @brief Get DC coefficients from a quantized image
* @param image Quantized image
* @param image_size Image size
* @param block_size Block size
* @return DC coefficients
*/
float* get_dc_coefficients(float* image, int image_size, int block_size)
{
    // ratio between image and block
    int N = image_size / block_size;

    float* dc_coeff = new float[N * N];

    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++) {
            dc_coeff[x * N + y] = image[x * block_size * image_size + y * block_size];
        }
    }
    return dc_coeff;
}

/**
* @brief Generate the 8x8 JPEG quantization weights for a given quality factor
* @param q_factor Quality factor
* @return Quantization weights matrix
*/
float* quantization_table(int q_factor)
{
    // ref: https://stackoverflow.com/questions/29215879/how-can-i-generalize-the-quantization-matrix-in-jpeg-compression

    float base_matrix[] = { 16, 11, 10, 16, 24, 40, 51, 61, \
                            12, 12, 14, 19, 26, 58, 60, 55, \
                            14, 13, 16, 24, 40, 57, 69, 56, \
                            14, 17, 22, 29, 51, 87, 80, 62, \
                            18, 22, 37, 56, 68, 109, 103, 77, \
                            24, 35, 55, 64, 81, 104, 113, 92, \
                            49, 64, 78, 87, 103, 121, 120, 101, \
                            72, 92, 95, 98, 112, 100, 103, 99 };
    int s = 0;

    if (q_factor < 50)
        s = 5000 / q_factor;
    else
        s = 200 - 2 * q_factor;

    float* quantization = new float[8 * 8];

    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            int value = floor((base_matrix[x * 8 + y] * s + 50) / 100);
            if (value == 0)
                quantization[x * 8 + y] = 1.0f; // prevent divide by 0
            else
                quantization[x * 8 + y] = (float) value;

        }
    }
    return quantization;
}

/**
* @brief Generate normalized discrete cosine transform (DCT) basis vectors
* @param size Length of the vectors
* @return DCT basis vectors
*/
float* generate_dct_basis(int size)
{
    // create DCT basis vectors matrix
    float* dct_basis = new float[size * size];

    for (int k = 0; k < size; k++) {
        for (int n = 0; n < size; n++) {
            dct_basis[k * size + n] = cos(M_PI / size * (n + 0.5) * k);
        }
    }

    // normalize
    // multiply X0 by 1/sqrt(N)
    // k = 0
    for (int n = 0; n < size; n++) {
        dct_basis[n] = dct_basis[n] / sqrt(size);
    }

    // multiply other elements by sqrt(2/N)
    for (int k = 1; k < size; k++) {
        for (int n = 0; n < size; n++) {
            dct_basis[k * size + n] = dct_basis[k * size + n] * sqrt(2) / sqrt(size);
        }
    }

    return dct_basis;
}

/**
* @brief Produce DCT coefficients of a given image using a given block size
* @param image Image file
* @param image_size Image size
* @param dct_basis DCT basis vectors
* @param block_size Block size
* @return DCT coefficients of the given image
*/
float* block_transform(float* image, int image_size, float* dct_basis, int block_size)
{
    float* coefficients = new float[image_size * image_size];

    for (int x = 0; x < image_size; x += block_size) {
        for (int y = 0; y < image_size; y += block_size) {
            // split image into small blocks
            float* block = new float[block_size * block_size];

            for (int a = 0; a < block_size; a++) {
                for (int b = 0; b < block_size; b++) {
                    block[a * block_size + b] = image[(x + a) * image_size + (y + b)];
                }
            }

            // DCT coefficients
            // reassemble the image after calculating DCT
            float* block_coeff = transform(block, dct_basis, block_size);

            for (int a = 0; a < block_size; a++) {
                for (int b = 0; b < block_size; b++) {
                    coefficients[(x + a) * image_size + (y + b)] = block_coeff[a * block_size + b];
                }
            }

            delete[] block_coeff;
        }
    }

    return coefficients;
}

/**
* @brief Generate the quantized image file
* @param image Image file
* @param image_size Image size
* @param quant_table Quantization weights
* @param block_size Block size
* @return Quantized image
*/
float* quantize(float* image, int image_size, float* quant_table, int block_size)
{
    float* quantized = new float[image_size * image_size];

    for (int x = 0; x < image_size; x += block_size) {
        for (int y = 0; y < image_size; y += block_size) {

            // split image into small blocks
            float* block = new float[block_size * block_size];
            for (int a = 0; a < block_size; a++) {
                for (int b = 0; b < block_size; b++) {
                    quantized[(x + a) * image_size + (y + b)] = (floor)((image[(x + a) * image_size + (y + b)] + quant_table[a * block_size + b] / 2) / quant_table[a * block_size + b]);
                }
            }
            delete[] block;
        }
    }

    return quantized;
   
}

/**
* @brief Dequantize (or Inverse quantize) an quantized image using the given quantization weights
* @param quantized_image Quantized image
* @param image_size Image size
* @param quant_table Quantization weights
* @param block_size Block size
* @return Dequantized image
*/
float* dequantize(float* quantized_image, int image_size, float* quant_table, int block_size)
{
    float* dequantize = new float[image_size * image_size];

    for (int x = 0; x < image_size; x += block_size) {
        for (int y = 0; y < image_size; y += block_size) {

            // split image into small blocks
            //float* block = new float[block_size * block_size];
            for (int a = 0; a < block_size; a++) {
                for (int b = 0; b < block_size; b++) {
                    dequantize[(x + a) * image_size + (y + b)] = quantized_image[(x + a) * image_size + (y + b)] * quant_table[a * block_size + b];
                }
            }
            //delete[] block;
        }
    }

    return dequantize;
}

/**
* @brief Encode a given image file (Produce the quantized image)
* @param image Image file
* @param image_size Image size
* @param quant_table Quantization weights
* @param block_size Block size
* @return Encoded (quantized) image
*/
float* encode(float* image, int image_size, float* quant_table, int block_size)
{
    // return a zero matrix if image size is not divisible by the block size
    if (image_size % block_size != 0)
    {
        float* zero_matrix = new float[image_size * image_size];

        for (int i = 0; i < image_size; i++)
            for (int j = 0; j < image_size; j++)
                zero_matrix[i * image_size + j] = 0.0f;

        return zero_matrix;
    }

    float* dct_basis = generate_dct_basis(block_size);
    float* dct_coeff = block_transform(image, image_size, dct_basis, block_size);
    delete[] dct_basis;
    float* quantized_coeff = quantize(dct_coeff, image_size, quant_table, block_size);
    delete[] dct_coeff;

    return quantized_coeff;
}

/**
* @brief Decode an encoded (quantized) image file
* @param quantized_coeff Quantized (encoded) image 
* @param image_size Image size
* @param quant_table Quantization weights
* @param block_size Block size
* @return Decoded image
*/
float* decode(float* quantized_coeff, int image_size, float* quant_table, int block_size)
{
    if (image_size % block_size != 0)
    {
        float* zero_matrix = new float[image_size * image_size];

        for (int i = 0; i < image_size; i++)
            for (int j = 0; j < image_size; j++)
                zero_matrix[i * image_size + j] = 0.0f;

        return zero_matrix;
    }

    float* dct_basis = generate_dct_basis(block_size);
    float* idct_basis = transpose(dct_basis, block_size);
    delete[] dct_basis;
    float* dequantized_coeff = dequantize(quantized_coeff, image_size, quant_table, block_size);
    float* reconstructed_image = block_transform(dequantized_coeff, image_size, idct_basis, block_size);
    delete[] idct_basis;
    delete[] dequantized_coeff;

    return reconstructed_image;
}

/**
* @brief Produce the resultant image using a given quantization weights and a block size
* @param image Image file
* @param image_size Image size
* @param quant_table Quantization weights
* @param block_size Block size
* @return Reconstructed image
*/
float* approximate(float* image, int image_size, float* quant_table, int block_size)
{
    if (image_size % block_size != 0)
    {
        float* zero_matrix = new float[image_size * image_size];

        for (int i = 0; i < image_size; i++)
            for (int j = 0; j < image_size; j++)
                zero_matrix[i * image_size + j] = 0.0f;

        return zero_matrix;
    }

    float* dct_basis = generate_dct_basis(block_size);
    float* idct_basis = transpose(dct_basis, block_size);

    float* dct_coeff = new float[image_size * image_size];
    float* quantized_image = new float[image_size * image_size];
    float* dequantized_image = new float[image_size * image_size];
    float* reconstructed_image = new float[image_size * image_size];

    float* dct32x32_dct = new float[image_size * image_size];
    float* dct32x32_quantize = new float[image_size * image_size];

    int N = image_size / block_size;


    for (int x = 0; x < image_size; x += block_size) {
        for (int y = 0; y < image_size; y += block_size) {


            // split image into smaller blocks
            float* block = new float[block_size * block_size];

            for (int a = 0; a < block_size; a++) {
                for (int b = 0; b < block_size; b++) {
                    block[a * block_size + b] = image[(x + a) * image_size + (y + b)];
                }
            }

            // calculate DCT coefficients
            float* transformed_block = transform(block, dct_basis, block_size);

            // 8x8 contiguos blocks of dct coefficients
            for (int a = 0; a < block_size; a++) {
                for (int b = 0; b < block_size; b++) {
                    dct_coeff[(x + a) * image_size + (y + b)] = transformed_block[a * block_size + b];
                }
            }

            // 32x32 blocks of interleaved dct coefficients
            int r = x / block_size;
            int c = y / block_size;
            
            for (int a = 0; a < block_size; a++) {
                for (int b = 0; b < block_size; b++) {
                    dct32x32_dct[(r + a * N) * image_size + (c + b * N)] = transformed_block[a * block_size + b];
                }
            }

            // quantize
            float* quantized_block = new float[block_size * block_size];

            for (int a = 0; a < block_size; a++) {
                for (int b = 0; b < block_size; b++) {
                    quantized_block[a * block_size + b] = (floor)((transformed_block[a * block_size + b] + quant_table[a * block_size + b] / 2) / quant_table[a * block_size + b]);
                }
            }

            // reassemble the original image after quantization
            for (int a = 0; a < block_size; a++) {
                for (int b = 0; b < block_size; b++) {
                    quantized_image[(x + a) * image_size + (y + b)] = quantized_block[a * block_size + b];
                    // 32x32 blocks of interleaved dct coefficients
                    dct32x32_quantize[(r + a * 32) * 256 + (c + b * 32)] = quantized_block[a * block_size + b];
                }
            }

            // dequantize
            float* dequantized_block = new float[block_size * block_size];

            for (int a = 0; a < block_size; a++) {
                for (int b = 0; b < block_size; b++) {
                    dequantized_block[a * block_size + b] = (quantized_block[a * block_size + b] * quant_table[a * block_size + b]);
                }
            }

            // reassemble the original image after dequantization
            for (int a = 0; a < block_size; a++) {
                for (int b = 0; b < block_size; b++) {
                    dequantized_image[(x + a) * image_size + (y + b)] = dequantized_block[a * block_size + b];
                }
            }

            // IDCT
            float* reconstructed_block = transform(dequantized_block, idct_basis, block_size);

            // reassemble the original image
            for (int a = 0; a < block_size; a++) {
                for (int b = 0; b < block_size; b++) {
                    reconstructed_image[(x + a) * image_size + (y + b)] = reconstructed_block[a * block_size + b];
                }
            }

            delete[] block;
            delete[] transformed_block;
            delete[] quantized_block;
            delete[] dequantized_block;
            delete[] reconstructed_block;

        }
    }

    store(dct_coeff, "dct_coeff.raw", image_size);
    store(quantized_image, "quantized_image.raw", image_size);
    store(dequantized_image, "dequantized_image.raw", image_size);
    store(reconstructed_image, "reconstructed_image.raw", image_size);

    store(dct32x32_dct, "dct32x32_dct.raw", image_size);
    store(dct32x32_quantize, "dct32x32_quantized.raw", image_size);

    delete[] dct_basis;
    delete[] idct_basis;
    delete[] dct_coeff;
    delete[] quantized_image;
    delete[] dequantized_image;
    delete[] dct32x32_dct;
    delete[] dct32x32_quantize;

    return reconstructed_image;
}

/**
* @brief Add noise to an image file
* @param image Image data
* @param noise Noise data
* @return Noise added image
*/
float* add_noise(float* image, float* noise)
{
    float* result = new float[256 * 256];

    for (int y = 0; y < 256; y++) {
        for (int x = 0; x < 256; x++) {
            result[y * 256 + x] = image[y * 256 + x] + noise[y * 256 + x];
        }
    }

    return result;
}

/**
* @brief Generate uniform-distributed random noise with given max and min values
* @param size Size of the noise image
* @param lower Lower boundary of the random noise (Min)
* @param upper Upper boundary of the random noise (Max)
* @return Generated noise data
*/
float* generate_uniform_distributed_noise(int size, float lower, float upper)
{
    random_device rd;  // obtain a seed for the random number engine
    mt19937 gen(rd()); // Standard mersenne_twister_engine
    uniform_real_distribution<float> distribution(lower, upper);

    float* uniform_image = new float[size * size];

    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            uniform_image[y * size + x] = distribution(gen);
        }
    }

    return uniform_image;

}

/**
* @brief Generate Gaussian-distributed random noise with given mean and standard deviation
* @param size Size of the noise image
* @param mean Mean of the Gaussian-distribution
* @param std Standard deviation of the Gaussian-distribution
* @return Generated noise data
*/
float* generate_gaussian_distributed_noise(int size, float mean, float std)
{
    random_device rd;  // obtain a seed for the random number engine
    mt19937 gen(rd()); // Standard mersenne_twister_engine
    normal_distribution<float> dis(mean, std);

    float* gaussian = new float[size * size];

    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            gaussian[y * size + x] = dis(gen);
        }
    }

    return gaussian;

}

/**
* @brief Set values of DCT coefficients below the threshold to zero
* @param dct_coefficients DCT coefficients
* @param threshold Threashold value
* @return Thresholded DCT coefficients matrix
*/
float* threshold(float* dct_coefficients, float threshold)
{
    float* result = new float[256 * 256];

    for (int x = 0; x < 256; x++) {
        for (int y = 0; y < 256; y++) {
            // if absolute value of coeffient is less than threshold, set it to 0
            if (abs(dct_coefficients[x * 256 + y]) < threshold)
                result[x * 256 + y] = 0.0;
            // if not, keep the value
            else
                result[x * 256 + y] = dct_coefficients[x * 256 + y];
        }
    }

    return result;
}

/**
* @brief Produce DCT coefficients from a given image
* @param image Image file
* @param coefficients DCT basis vectors
* @param size Size of the image
* @return DCT coefficients of the given image
*/
float* transform(float* image, float* coefficients, int size)
{
    // transpose of DCT basis matrix
    float* trans_coefficients = transpose(coefficients, size);

    float* AX = matmul(coefficients, image, size);
    float* AXAT = matmul(AX, trans_coefficients, size);

    return AXAT;
}

/**
* @brief Multiply two matrices
* @param matrix1 First matrix
* @param matrix2 Second matrix
* @param size Size of the matrices
* @return Result of the multiplication
*/
float* matmul(float *matrix1, float *matrix2, int size)
{
    // create result matrix
    float* result = new float[size * size];

    // initialize all to 0
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            result[y * size + x] = 0;
        }
    }

    // matrix multiplication
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            for (int k = 0; k < size; k++)
            {
                result[i * size + j] += matrix1[i * size + k] * matrix2[k * size + j];
            }
        }
    }

    return result;

}

/**
* @brief Transpose a given matrix
* @param inputMatrix Input matrix
* @param size Size of the matrix
* @return Transposed matrix
*/
float* transpose(float *inputMatrix, int size)
{
    float* result = new float[size * size];

    for (int x = 0; x < size; x++) {
        for (int y = 0; y < size; y++) {
            result[y * size + x] = inputMatrix[x * size + y];
        }
    }

    return result;
}

/**
* @brief Save an image file
* @param image Image file data
* @param filename Name of the output file
* @param size Size of the image
*/
void store(float* image, string filename, int size)
{
    ofstream ofs(filename, ios::out | ios::binary);
    ofs.write((char*)&image[0], sizeof(float)* size * size);
    ofs.close();

}

/**
* @brief Load an image file
* @param filename Input file name
* @return Image data
*/
float *load(string filename)
{
    ifstream ifs(filename, ifstream::binary);

    if (ifs) 
    {
        // get length of file
        ifs.seekg(0, ifs.end);
        int length = ifs.tellg();
        ifs.seekg(0, ifs.beg);

        // buffer to store the file
        char* buffer = new char[length];

        // read data as a block
        ifs.read(buffer, length);
        ifs.close();

        return (float*)buffer;
    }
    else 
    {  
        std::cout << "Can't open file " << filename << std::endl;
        return nullptr;
    }
}

/**
* @brief Calculate the mean squared error between two images
* @param image1 First image
* @param image2 Second image
* @param size Size of the image files
* @return Mean squared error
*/
float mse(float* image1, float *image2, int size)
{
    float error = 0;

    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            error += pow(image1[y * size + x] - image2[y * size + x], 2);
        }
    }
    float mse = error / (size * size);
    cout << "Mean squared Error: " << mse << endl;

    return mse;
}

/**
* @brief Calculate the Peak signal-to-noise ratio between two images
* @param image1 First image
* @param image2 Second image
* @param size Size of the image files
* @return Peak signal-to-noise ratio
*/
float psnr(float* image1, float* image2, int size, int max)
{
    float error = mse(image1, image2, size);
    float ratio = 20 * log10(max / sqrt(error));
    cout << "Peak Signal-to-Noise Ratio: " << ratio << endl;
    return ratio;
}
