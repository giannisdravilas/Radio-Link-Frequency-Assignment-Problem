/////////////////////////////////////////////////////////////////////////////
//
// Υλοποίηση του ADT Map μέσω Hash Table με open addressing (linear probing)
//
/////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>

#include "ADTMap.h"

#include "ADTList.h"

// Το μέγεθος του Hash Table ιδανικά θέλουμε να είναι πρώτος αριθμός σύμφωνα με την θεωρία.
// Η παρακάτω λίστα περιέχει πρώτους οι οποίοι έχουν αποδεδιγμένα καλή συμπεριφορά ως μεγέθη.
// Κάθε re-hash θα γίνεται βάσει αυτής της λίστας. Αν χρειάζονται παραπάνω απο 1610612741 στοχεία, τότε σε καθε rehash διπλασιάζουμε το μέγεθος.
int prime_sizes[] = {53, 97, 193, 389, 769, 1543, 3079, 6151, 12289, 24593, 49157, 98317, 196613, 393241,
	786433, 1572869, 3145739, 6291469, 12582917, 25165843, 50331653, 100663319, 201326611, 402653189, 805306457, 1610612741};

// Χρησιμοποιούμε open addressing, οπότε σύμφωνα με την θεωρία, πρέπει πάντα να διατηρούμε
// τον load factor του  hash table μικρότερο ή ίσο του 0.9, για να έχουμε αποδoτικές πράξεις
#define MAX_LOAD_FACTOR 0.9

// Δομή του κάθε κόμβου που έχει το hash table (με το οποίο υλοιποιούμε το map)
struct map_node{
	Pointer key;		// Το κλειδί που χρησιμοποιείται για να hash-αρουμε
	Pointer value;  	// Η τιμή που αντιστοιχίζεται στο παραπάνω κλειδί
};

// Δομή του Map (περιέχει όλες τις πληροφορίες που χρεαζόμαστε για το HashTable)
struct map {
	List* array;				// Ο πίνακας που θα χρησιμοποιήσουμε για το map (remember, φτιάχνουμε ένα hash table)
	int capacity;				// Πόσο χώρο έχουμε δεσμεύσει.
	int size;					// Πόσα στοιχεία έχουμε προσθέσει
	CompareFunc compare;		// Συνάρτηση για σύγκρηση δεικτών, που πρέπει να δίνεται απο τον χρήστη
	HashFunc hash_function;		// Συνάρτηση για να παίρνουμε το hash code του κάθε αντικειμένου.
	DestroyFunc destroy_key;	// Συναρτήσεις που καλούνται όταν διαγράφουμε έναν κόμβο απο το map.
	DestroyFunc destroy_value;
};


Map map_create(CompareFunc compare, DestroyFunc destroy_key, DestroyFunc destroy_value) {
	// Δεσμεύουμε κατάλληλα τον χώρο που χρειαζόμαστε για το hash table
	Map map = malloc(sizeof(*map));
	map->capacity = prime_sizes[0];
	map->array = malloc(map->capacity * sizeof(List*));

	for(int i = 0; i < map->capacity; i++){
		map->array[i] = list_create(NULL);
	}

	map->size = 0;
	map->compare = compare;
	map->destroy_key = destroy_key;
	map->destroy_value = destroy_value;

	return map;
}

// Επιστρέφει τον αριθμό των entries του map σε μία χρονική στιγμή.
int map_size(Map map) {
	return map->size;
}

// Συνάρτηση για την επέκταση του Hash Table σε περίπτωση που ο load factor μεγαλώσει πολύ.
static void rehash(Map map) {
	// Αποθήκευση των παλιών δεδομένων
	int old_capacity = map->capacity;
	List* old_array = map->array;

	// Βρίσκουμε τη νέα χωρητικότητα, διασχίζοντας τη λίστα των πρώτων ώστε να βρούμε τον επόμενο. 
	int prime_no = sizeof(prime_sizes) / sizeof(int);	// το μέγεθος του πίνακα
	for (int i = 0; i < prime_no; i++) {					// LCOV_EXCL_LINE
		if (prime_sizes[i] > old_capacity) {
			map->capacity = prime_sizes[i]; 
			break;
		}
	}
	// Αν έχουμε εξαντλήσει όλους τους πρώτους, διπλασιάζουμε
	if (map->capacity == old_capacity)					// LCOV_EXCL_LINE
		map->capacity *= 2;								// LCOV_EXCL_LINE

	// Δημιουργούμε ένα μεγαλύτερο hash table
	map->array = malloc(map->capacity * sizeof(List*));

	for(int i = 0; i < map->capacity; i++){
		map->array[i] = list_create(NULL);
	}

	// Τοποθετούμε όλα τα entries στο νέο hash table, αποδεσμεύοντας παράλληλα όποια μνήμη
	// δεν χρειαζόμαστε πλέον από το παλιό hash table
	map->size = 0;
	for (int i = 0; i < old_capacity; i++){
		for(ListNode node = list_first(old_array[i]); node != LIST_EOF; node = list_next(old_array[i], node)){
			MapNode node_map = list_node_value(old_array[i], node);
			map_insert(map, node_map->key, node_map->value);
			free(node_map);
		}
		list_destroy(old_array[i]);
	}

	//Αποδεσμεύουμε τον παλιό πίνακα ώστε να μην έχουμε leaks
	free(old_array);
}

// Εισαγωγή στο hash table του ζευγαριού (key, item). Αν το key υπάρχει,
// ανανέωση του με ένα νέο value, και η συνάρτηση επιστρέφει true.

void map_insert(Map map, Pointer key, Pointer value) {

	bool already_in_map = false;

	//Θα τοποθετήσουμε το νέο key στη λίστα που βρίσκεται στη θέση που αυτό κάνει hash
	uint pos = map->hash_function(key) % map->capacity;

	ListNode node;
	MapNode node_map;

	//Βρίσκουμε αν το key υπάρχει ήδη στη συγκεκριμένη λίστα
	for (node = list_first(map->array[pos]); node != LIST_EOF; node = list_next(map->array[pos], node)){
		if(node != NULL){
			node_map = list_node_value(map->array[pos], node);
			if (map->compare(node_map->key, key) == 0) {
				already_in_map = true;
				break;
			}
		}
	}

	if (already_in_map) {
		// Αν αντικαθιστούμε παλιά key/value, τa κάνουμε destropy
		if (node_map->key != key && map->destroy_key != NULL)
			map->destroy_key(node_map->key);

		if (node_map->value != value && map->destroy_value != NULL)
			map->destroy_value(node_map->value);

		// Προσθήκη τιμών στον κόμβο
		node_map->key = key;
		node_map->value = value;

	} else {
		// Νέο στοιχείο, αυξάνουμε τα συνολικά στοιχεία του map
		map->size++;
		MapNode node_insert = malloc(sizeof(*node_insert));
		// Προσθήκη τιμών στον κόμβο
		node_insert->key = key;
		node_insert->value = value;
		list_insert_next(map->array[pos], node, node_insert);
	}


	// Αν με την νέα εισαγωγή ξεπερνάμε το μέγιστο load factor, πρέπει να κάνουμε rehash
	float load_factor = (float)map->size / map->capacity;
	if (load_factor > MAX_LOAD_FACTOR)
		rehash(map);
}

// Διαγραφή από το Hash Table του κλειδιού με τιμή key
bool map_remove(Map map, Pointer key) {

	// Στη θέση που κάνει hash το key θα βρίσκεται η λίστα που το περιέχει
	uint pos = map->hash_function(key) % map->capacity;
	
	ListNode node_previous = LIST_BOF;
	MapNode node_map;
	bool found = false;

	//Ψάχνουμε στη λίστα για να δούμε αν το key όντως υπάρχει
	for(ListNode node = list_first(map->array[pos]); node != LIST_EOF; node = list_next(map->array[pos], node)){
		node_map = list_node_value(map->array[pos], node);
		if(map->compare(node_map->key, key) == 0){
			found = true;
			break;
		}
		node_previous = node;
	}

	//Αν υπάρχει, τότε το διαγράφουμε
	if(found == true){
		// destroy
		if (map->destroy_key != NULL)
			map->destroy_key(node_map->key);
		if (map->destroy_value != NULL)
			map->destroy_value(node_map->value);

		free(node_map);

		list_remove_next(map->array[pos], node_previous);

		map->size--;

		return true;
	}

	return false;
}

// Αναζήτηση στο map, με σκοπό να επιστραφεί το value του κλειδιού που περνάμε σαν όρισμα.

Pointer map_find(Map map, Pointer key) {
	MapNode node = map_find_node(map, key);
	if (node != MAP_EOF)
		return node->value;
	else
		return NULL;
}

DestroyFunc map_set_destroy_key(Map map, DestroyFunc destroy_key) {
	DestroyFunc old = map->destroy_key;
	map->destroy_key = destroy_key;
	return old;
}

DestroyFunc map_set_destroy_value(Map map, DestroyFunc destroy_value) {
	DestroyFunc old = map->destroy_value;
	map->destroy_value = destroy_value;
	return old;
}

// Απελευθέρωση μνήμης που δεσμεύει το map
void map_destroy(Map map) {

	//Για κάθε map_node
	for (int i = 0; i < map->capacity; i++) {

		//Βρίσκουμε τη λίστα του και τη διαγράφουμε, μαζί με όση μνήμη αυτή δεσμεύει
		for(ListNode node = list_first(map->array[i]); node != LIST_EOF; node = list_next(map->array[i], node)){
			MapNode node_map = list_node_value(map->array[i], node);
			if (map->destroy_key != NULL)
				map->destroy_key(node_map->key);
			if (map->destroy_value != NULL)
				map->destroy_value(node_map->value);
			free(node_map);
		}
		list_destroy(map->array[i]);
	}

	free(map->array);
	free(map);
}

/////////////////////// Διάσχιση του map μέσω κόμβων ///////////////////////////

MapNode map_first(Map map) {

	//Ξεκινάμε ψάχνοντας για κάθε map_node
	for(int i = 0; i < map->capacity; i++){
		
		//Ένα ένα τα στοιχεία της λίστας του (αν υπάρχουν)
		for(ListNode node = list_first(map->array[i]); node != LIST_EOF; node = list_next(map->array[i], node)){

			//Επιστρέφουμε το πρώτο στοιχείο που θα συναντήσουμε
			if(node != NULL){
				return list_node_value(map->array[i], node);
			}
		}
	}
	return MAP_EOF;
}

MapNode map_next(Map map, MapNode node) {

	//Βρίσκουμε πού κάνει hash το key του κόμβου που δόθηκε
	uint pos = map->hash_function(node->key) % map->capacity;

	ListNode node_list;
	bool exists = false;

	//Αρχικά ψάχνουμε στην λίστα που αντιστοιχεί στο σημείο που κάνει το hash το key, αν ο node υπάρχει
	for(node_list = list_first(map->array[pos]); node_list != LIST_EOF; node_list = list_next(map->array[pos], node_list)){

		MapNode node_map = list_node_value(map->array[pos], node_list);
		//Αν τον βρούμε, τότε σταματάμε
		if(node_map == node){
			exists = true;
			break;
		}
	}

	//Αν ο node υπάρχει
	if(exists){

		//Ξεκινάμε από τον επόμενό του στην ίδια λίστα
		for(ListNode node_list_next = list_next(map->array[pos], node_list); node_list_next != LIST_EOF; node_list_next = list_next(map->array[pos], node_list_next)){
			
			//Αν μπούμε στην επανάληψη τότε υπάρχει επόμενος κόμβος και άρα επιστρέφουμε αυτόν
			return list_node_value(map->array[pos], node_list_next);
		}

		//Αλλιώς ξεκινάμε να ψάχνουμε κάθε επόμενη λίστα μέχρι το τέλος του map->array
		for(int i = pos+1; i < map->capacity; i++){

			//Για κάθε στοιχείο της λίστας
			for(node_list = list_first(map->array[i]); node_list != LIST_EOF; node_list = list_next(map->array[i], node_list)){
				
				//Αν μπούμε στην επανάληψη τότε η λίστα δεν είναι κενή, δηλαδή έχει στοιχείο, άρα το επιστρέφουμε
				return list_node_value(map->array[i], node_list);
			}
		}
	}
	return MAP_EOF;
}

Pointer map_node_key(Map map, MapNode node) {
	return node->key;
}

Pointer map_node_value(Map map, MapNode node) {
	return node->value;
}

MapNode map_find_node(Map map, Pointer key) {

	if(key){
		//Βρίσκουμε τη θέση που κάνει hash το key
		uint pos = map->hash_function(key) % map->capacity;

		//Ψάχνουμε κάθε στοιχείο της λίστας που αντιστοιχεί στη συγκεκριμένη θέση
		for(ListNode node = list_first(map->array[pos]); node != LIST_EOF; node = list_next(map->array[pos], node)){
			MapNode node_map = list_node_value(map->array[pos], node);

			//Αν το key του τρέχοντος στοιχείου ισούται με το key που ψάχνουμε, τότε επιστρέφουμε τον τρέχοντα κόμβο
			if(map->compare(node_map->key, key) == 0){
				return node_map;
			}
		}
	}
	return MAP_EOF;
}

// Αρχικοποίηση της συνάρτησης κατακερματισμού του συγκεκριμένου map.
void map_set_hash_function(Map map, HashFunc func) {
	map->hash_function = func;
}

uint hash_string(Pointer value) {
	// djb2 hash function, απλή, γρήγορη, και σε γενικές γραμμές αποδοτική
    uint hash = 5381;
    for (char* s = value; *s != '\0'; s++)
		hash = (hash << 5) + hash + *s;			// hash = (hash * 33) + *s. Το foo << 5 είναι γρηγορότερη εκδοχή του foo * 32.
    return hash;
}

uint hash_int(Pointer value) {
	return *(int*)value;
}

uint hash_pointer(Pointer value) {
	return (size_t)value;				// cast σε sizt_t, που έχει το ίδιο μήκος με έναν pointer
}


// Για χρήση στο DiseaseMonitor.c, δεν επηρεάζουν το interface του ADTMap

struct map_count{
    String disease;
    String country;
    String date;
};

typedef struct map_count* MapCount;

uint hash_struct_disease_country_date(Pointer value){
	return hash_string(((MapCount)(value))->date);
}