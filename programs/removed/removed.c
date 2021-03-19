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
        map_insert(map_new, new_codes[i], NULL);
    }

    for(MapNode node = map_first(map_site); node != MAP_EOF; node = map_next(map_site, node)){
        MapNode found = map_find_node(map_new, map_node_key(map_site, node));
        if(found == NULL && strcmp(map_node_value(map_site, node), "1") == 0){
            printf("%s\n", (char*)map_node_key(map_site, node));
        }
    }

    for(int i = 0; i < MAX_ITEMS; i++){
        free(codes[i]);
        free(current_state[i]);
        free(new_codes[i]);
    }
    free(codes);
    free(current_state);
    free(new_codes);

    map_destroy(map_site);
    map_destroy(map_new);
}