#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ITEMS 10000

int main(void){

    printf("give 'ΣΩΣΤΟ', 'ΛΑΘΟΣ':\n");
    char** cell;
    cell = malloc(MAX_ITEMS*sizeof(char*));
    for(int i = 0; i < MAX_ITEMS; i++){
        cell[i] = malloc(10*sizeof(char));
        strcpy(cell[i], "");
    }
    int i = 0;
    scanf("%s ", cell[i]);
    while(strcmp(cell[i], "0") != 0){
        i++;
        scanf("%s ", cell[i]);
    }

    for(int i = 0; i < MAX_ITEMS; i++){
        if(strcmp(cell[i], "") == 0){
            break;
        }
        char* str = malloc(3*sizeof(char));
        strncpy(str, cell[i], 2);
        // printf("str is %s\n", str);
        if(strcmp(str, "Λ") == 0){
            printf("ΛΆΘΟΣ\n");
        }else if(strcmp(str, "Σ") == 0){
            printf("\n");
        }
    }
}