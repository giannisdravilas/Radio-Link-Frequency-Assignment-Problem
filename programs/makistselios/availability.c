#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ADTMap.h"

#define MAX_ITEMS 20000
#define CHARACTERS_COMPARE 2

int compare_strings(Pointer a, Pointer b){
    return strcmp(a, b);
}

int main(void){

    char* not_available = malloc(10*sizeof(char));
    strcpy(not_available, "ΕΞ");

    char* waitingel = malloc(10*sizeof(char));
    strcpy(waitingel, "ΑΝ");

    char* waitingen = malloc(10*sizeof(char));
    strcpy(waitingen, "ANAM");

    char* available = malloc(10*sizeof(char));
    strcpy(available, "ΔΙ");

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

    printf("give current site state (0/1) and stop with '2':\n");
    char** current_state;
    current_state = malloc(MAX_ITEMS*sizeof(char*));
    for(int i = 0; i < MAX_ITEMS; i++){
        current_state[i] = malloc(2*sizeof(char));
        strcpy(current_state[i], "");
    }
    i = 0;
    scanf("%s ", current_state[i]);
    while(strcmp(current_state[i], "2") != 0){
        i++;
        scanf("%s ", current_state[i]);
    }

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

    printf("give new state and stop with '0':\n");
    char** new_state;
    new_state = malloc(MAX_ITEMS*sizeof(char*));
    for(int i = 0; i < MAX_ITEMS; i++){
        new_state[i] = malloc(100*sizeof(char));
        strcpy(new_state[i], "");
    }
    i = 0;

    char* str = malloc(100*sizeof(char));
    while((fgets(str, 100, stdin)) && strcmp(str, "0") != 0){
        strncpy(new_state[i], str, CHARACTERS_COMPARE+2);
        i++;
    }

    char** changed_state;
    changed_state = malloc(MAX_ITEMS*sizeof(char*));
    for(int i = 0; i < MAX_ITEMS; i++){
        changed_state[i] = malloc(100*sizeof(char));
        strcpy(changed_state[i], "");
    }

    char** new_products;
    new_products = malloc(MAX_ITEMS*sizeof(char*));
    for(int i = 0; i < MAX_ITEMS; i++){
        new_products[i] = malloc(100*sizeof(char));
        strcpy(new_products[i], "");
    }

    int changed_state_index = 0;
    int new_products_index = 0;

    Map map_site = map_create(compare_strings, NULL, NULL);
    map_set_hash_function(map_site, hash_string);
    for(int i = 0; i < MAX_ITEMS; i++){
        if(strcmp(codes[i], "0") == 0){
            break;
        }
        map_insert(map_site, codes[i], current_state[i]);
    }

    Map map_new = map_create(compare_strings, NULL, NULL);
    map_set_hash_function(map_new, hash_string);
    for(int i = 0; i < MAX_ITEMS; i++){
        if(strcmp(new_codes[i], "0") == 0)
            break;
        map_insert(map_new, new_codes[i], new_state[i]);
    }

    for(MapNode node = map_first(map_new); node != MAP_EOF; node = map_next(map_new, node)){
        if(strcmp((String)map_node_value(map_new, node), not_available) != 0 && strcmp((String)map_node_value(map_new, node), waitingen) != 0 && strcmp((String)map_node_value(map_new, node), waitingel) != 0){
            MapNode found = map_find_node(map_site, map_node_key(map_new, node));
            if(found){
                if(strcmp(map_node_value(map_site, found), "0") == 0){
                    strcpy(changed_state[changed_state_index], map_node_key(map_site, found));
                    changed_state_index++;
                }
            }else if(strcmp(map_node_value(map_new, node), available) == 0){
                strcpy(new_products[new_products_index], map_node_key(map_new, node));
                new_products_index++;
            }
        }else if(strcmp(map_node_value(map_new, node), not_available) == 0 || strcmp(map_node_value(map_new, node), waitingen) == 0 || strcmp(map_node_value(map_new, node), waitingel) == 0){
            MapNode found = map_find_node(map_site, map_node_key(map_new, node));
            if(found){
                if(strcmp(map_node_value(map_site, found), "1") == 0){
                    strcpy(changed_state[changed_state_index], map_node_key(map_site, found));
                    changed_state_index++;
                }
            }
        }
    }

    printf("-----------------------------------\n");
    printf("\n\nchanged state (%d):\n\n", changed_state_index);
    printf("-----------------------------------\n");
    for(int i = 0; i < MAX_ITEMS; i++){
        if(strcmp(changed_state[i], "") == 0)
            break;
        printf("%s\n", changed_state[i]);
    }

    printf("-----------------------------------\n");
    printf("\n\nnew products (%d):\n\n", new_products_index);
    printf("-----------------------------------\n");
    for(int i = 0; i < MAX_ITEMS; i++){
        if(strcmp(new_products[i], "") == 0)
            break;
        printf("%s\n", new_products[i]);
    }

    printf("-----------------------------------\n");
    printf("\n\nlist of new states with respect to current product codes from site\n\n");
    printf("-----------------------------------\n");
    for(int i = 0; i < MAX_ITEMS; i++){
        if(strcmp(codes[i+1], "") == 0){
            break;
        }
        bool flag = false;
        for(int j = 0; j < MAX_ITEMS; j++){
            if(strcmp(changed_state[j], "") == 0){
                break;
            }
            if(strcmp(codes[i], changed_state[j]) == 0){
                if(strcmp(current_state[i], "0") == 0){
                    printf("1\n");
                }else if(strcmp(current_state[i], "1") == 0){
                    printf("0\n");
                }
                flag = true;
                break;
            }
        }
        if(flag == false)
            printf("%s\n", current_state[i]);
    }

    for(int i = 0; i < MAX_ITEMS; i++){
        free(codes[i]);
        free(current_state[i]);
        free(new_codes[i]);
        free(new_state[i]);
        free(new_products[i]);
        free(changed_state[i]);
    }
    free(codes);
    free(current_state);
    free(new_codes);
    free(new_state);
    free(new_products);
    free(changed_state);
    free(not_available);
    free(str);

    map_destroy(map_site);
    map_destroy(map_new);
}