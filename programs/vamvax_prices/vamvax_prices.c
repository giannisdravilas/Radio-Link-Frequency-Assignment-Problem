#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ADTMap.h"

#define MAX_ITEMS 20000

int compare_strings(Pointer a, Pointer b){
    return strcmp(a, b);
}

int main(void){

    printf("give site product codes and stop with '0':\n");
    char** codes;
    codes = malloc(MAX_ITEMS*sizeof(char*));
    for(int i = 0; i < MAX_ITEMS; i++){
        codes[i] = malloc(50*sizeof(char));
        strcpy(codes[i], "");
    }
    int i = 0;
    scanf("%s ", codes[i]);
    while(strcmp(codes[i], "0") != 0){
        i++;
        scanf("%s ", codes[i]);
    }

    printf("give current start prices and stop with '0':\n");
    double* current_price;
    current_price = malloc(MAX_ITEMS*sizeof(double));
    for(int i = 0; i < MAX_ITEMS; i++){
        current_price[i] = 0;
    }

    i = 0;
    scanf("%lf ", &current_price[i]);
    while(current_price[i] != 0){
        i++;
        scanf("%lf ", &current_price[i]);
    }

    printf("give current final prices and stop with '0':\n");
    double* current_final_price;
    current_final_price = malloc(MAX_ITEMS*sizeof(double));
    for(int i = 0; i < MAX_ITEMS; i++){
        current_final_price[i] = 0;
    }

    i = 0;
    scanf("%lf ", &current_final_price[i]);
    while(current_final_price[i] != 0){
        i++;
        scanf("%lf ", &current_final_price[i]);
    }

    
    // for(int i = 0; i < MAX_ITEMS; i++){
    //     if(strcmp(codes[i], "0") == 0){
    //         break;
    //     }
    //     printf("codes %s, %lf, %lf\n", codes[i], current_price[i], current_final_price[i]);
    // }

    printf("give new product codes and stop with '0':\n");
    char** new_codes;
    new_codes = malloc(MAX_ITEMS*sizeof(char*));
    for(int i = 0; i < MAX_ITEMS; i++){
        new_codes[i] = malloc(50*sizeof(char));
        strcpy(new_codes[i], "");
    }
    i = 0;
    scanf("%s ", new_codes[i]);
    while(strcmp(new_codes[i], "0") != 0){
        i++;
        scanf("%s ", new_codes[i]);
    }

    printf("give new prices and stop with '0':\n");
    double* new_price;
    new_price = malloc(MAX_ITEMS*sizeof(double));
    for(int i = 0; i < MAX_ITEMS; i++){
        new_price[i] = 0;
    }

    i = 0;
    scanf("%lf ", &new_price[i]);
    while(new_price[i] != 0){
        i++;
        scanf("%lf ", &new_price[i]);
    }

    printf("give discounts and stop with '100':\n");
    char** discount_char;
    discount_char = malloc(MAX_ITEMS*sizeof(char*));
    for(int i = 0; i < MAX_ITEMS; i++){
        discount_char[i] = malloc(10*sizeof(char));
        strcpy(discount_char[i], "");
    }
    i = 0;
    scanf("%s ", discount_char[i]);
    while(strcmp(discount_char[i], "100") != 0){
        i++;
        scanf("%s ", discount_char[i]);
        //printf("discount: code is %s, discount char is %s\n", new_codes[i], discount_char[i]);
    }

    double* discount = malloc(MAX_ITEMS*sizeof(double));
    for(int i = 0; i < MAX_ITEMS; i++){
        if(strcmp(new_codes[i+1], "") == 0){
            break;
        }
        discount[i] = 0;
        if(strcmp(discount_char[i], "") != 0 && strcmp(discount_char[i], "100") != 0){
            char* str = strtok(discount_char[i], "%");
            discount[i] = atof(str);
            //printf("here discount: code is %s, discount is %f\n", new_codes[i], discount[i]);
        }
    }

    Map map_start_prices = map_create(compare_strings, NULL, NULL);
    map_set_hash_function(map_start_prices, hash_string);
    for(int i = 0; i < MAX_ITEMS; i++){
        if(strcmp(new_codes[i], "0") == 0)
            break;
        map_insert(map_start_prices, new_codes[i], &new_price[i]);
        // printf("new code is %s, price is %f,  %f\n", new_codes[i], new_price[i], *(double*)map_find(map_start_prices, new_codes[i]));
    }

    Map map_final_prices = map_create(compare_strings, NULL, NULL);
    map_set_hash_function(map_final_prices, hash_string);
    for(int i = 0; i < MAX_ITEMS; i++){
        if(strcmp(new_codes[i], "0") == 0)
            break;
        double* final_price = malloc(sizeof(double));
        *(double*)final_price = (1-discount[i]/100) * new_price[i];
        map_insert(map_final_prices, new_codes[i], final_price);
        //printf("new codes %s, %lf, %f, %s\n", new_codes[i], *(double*)final_price, discount[i], discount_char[i]);
    }

    printf("-----------------------------------\n");
    printf("\n\nlist of start prices with respect to current product codes from site\n\n");
    printf("-----------------------------------\n");
    for(int i = 0; i < MAX_ITEMS; i++){
        if(strcmp(codes[i+1], "") == 0){
            break;
        }
        MapNode found = map_find_node(map_start_prices, codes[i]);
        if(found){
            double* price = (double*)map_node_value(map_start_prices, found);
            printf("%f\n", *(double*)price);
        }else{
            printf("%f\n", current_price[i]);
        }
    }

    printf("-----------------------------------\n");
    printf("\n\nlist of final prices with respect to current product codes from site\n\n");
    printf("-----------------------------------\n");
    for(int i = 0; i < MAX_ITEMS; i++){
        if(strcmp(codes[i+1], "") == 0){
            break;
        }
        MapNode found = map_find_node(map_final_prices, codes[i]);
        if(found){
            double* price = (double*)map_node_value(map_final_prices, found);
            MapNode found1 = map_find_node(map_start_prices, codes[i]);
            if(*(double*)price == 0 || *(double*)price == *(double*)map_node_value(map_start_prices, found1)){
                printf("\n");
            }else{
                printf("%f\n", *(double*)price);
            }
        }else{
            if(current_final_price[i] == 0){
                printf("\n");
            }else{
                printf("%f\n", current_final_price[i]);
            }
        }
    }

    for(MapNode node = map_first(map_final_prices); node != MAP_EOF; node = map_next(map_final_prices, node)){
        free(map_node_value(map_final_prices, node));
    }

    for(int i = 0; i < MAX_ITEMS; i++){
        free(codes[i]);
        free(new_codes[i]);
        free(discount_char[i]);
    }
    free(codes);
    free(current_price);
    free(current_final_price);
    free(new_codes);
    free(new_price);
    free(discount_char);
    free(discount);

    map_destroy(map_start_prices);
    map_destroy(map_final_prices);
}