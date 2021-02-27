import RLFA
import csp_dom_wdeg as csp
import time

if __name__ == '__main__':

    # For each instance, read the data and run the backtracking search with FC and MAC-AC3,
    # and the min-conflicts algorithm


    ######################
    ######## 2-f24 #######
    ######################

    print("\n2-f24\n")
    data = RLFA.Parsing("2-f24.txt")
    print("FC")
    start = time.time()
    fc_result, constraint_checks, visited_nodes = csp.backtracking_search(data, select_unassigned_variable=csp.dom_wdeg, inference=csp.forward_checking)
    end = time.time()
    print(fc_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)
    
    data = RLFA.Parsing("2-f24.txt")
    print("\nMAC")
    start = time.time()
    mac_result, constraint_checks, visited_nodes = csp.backtracking_search(data, select_unassigned_variable=csp.dom_wdeg, inference=csp.mac)
    end = time.time()
    print(mac_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)

    data = RLFA.Parsing("2-f24.txt")
    print("\nMin Conflicts")
    start = time.time()
    minConflicts_result, constraint_checks, visited_nodes = csp.min_conflicts(data)
    end = time.time()
    print(minConflicts_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)





    ######################
    ######## 2-f25 #######
    ######################

    print("\n2-f25\n")
    data = RLFA.Parsing("2-f25.txt")
    print("FC")
    start = time.time()
    fc_result, constraint_checks, visited_nodes = csp.backtracking_search(data, select_unassigned_variable=csp.dom_wdeg, inference=csp.forward_checking)
    end = time.time()
    print(fc_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)
    
    data = RLFA.Parsing("2-f25.txt")
    print("\nMAC")
    start = time.time()
    mac_result, constraint_checks, visited_nodes = csp.backtracking_search(data, select_unassigned_variable=csp.dom_wdeg, inference=csp.mac)
    end = time.time()
    print(mac_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)

    data = RLFA.Parsing("2-f25.txt")
    print("\nMin Conflicts")
    start = time.time()
    minConflicts_result, constraint_checks, visited_nodes = csp.min_conflicts(data)
    end = time.time()
    print(minConflicts_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)




    ######################
    ######## 3-f10 #######
    ######################

    print("\n3-f10\n")
    data = RLFA.Parsing("3-f10.txt")
    print("FC")
    start = time.time()
    fc_result, constraint_checks, visited_nodes = csp.backtracking_search(data, select_unassigned_variable=csp.dom_wdeg, inference=csp.forward_checking)
    end = time.time()
    print(fc_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)
    
    data = RLFA.Parsing("3-f10.txt")
    print("\nMAC")
    start = time.time()
    mac_result, constraint_checks, visited_nodes = csp.backtracking_search(data, select_unassigned_variable=csp.dom_wdeg, inference=csp.mac)
    end = time.time()
    print(mac_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)

    data = RLFA.Parsing("3-f10.txt")
    print("\nMin Conflicts")
    start = time.time()
    minConflicts_result, constraint_checks, visited_nodes = csp.min_conflicts(data)
    end = time.time()
    print(minConflicts_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)





    ######################
    ######## 3-f11 #######
    ######################

    print("\n3-f11\n")
    data = RLFA.Parsing("3-f11.txt")
    print("FC")
    start = time.time()
    fc_result, constraint_checks, visited_nodes = csp.backtracking_search(data, select_unassigned_variable=csp.dom_wdeg, inference=csp.forward_checking)
    end = time.time()
    print(fc_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)
    
    data = RLFA.Parsing("3-f11.txt")
    print("\nMAC")
    start = time.time()
    mac_result, constraint_checks, visited_nodes = csp.backtracking_search(data, select_unassigned_variable=csp.dom_wdeg, inference=csp.mac)
    end = time.time()
    print(mac_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)

    data = RLFA.Parsing("3-f11.txt")
    print("\nMin Conflicts")
    start = time.time()
    minConflicts_result, constraint_checks, visited_nodes = csp.min_conflicts(data)
    end = time.time()
    print(minConflicts_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)





    ######################
    ######## 6-w2 ########
    ######################

    print("\n6-w2\n")
    data = RLFA.Parsing("6-w2.txt")
    print("FC")
    start = time.time()
    fc_result, constraint_checks, visited_nodes = csp.backtracking_search(data, select_unassigned_variable=csp.dom_wdeg, inference=csp.forward_checking)
    end = time.time()
    print(fc_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)
    
    data = RLFA.Parsing("6-w2.txt")
    print("\nMAC")
    start = time.time()
    mac_result, constraint_checks, visited_nodes = csp.backtracking_search(data, select_unassigned_variable=csp.dom_wdeg, inference=csp.mac)
    end = time.time()
    print(mac_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)

    data = RLFA.Parsing("6-w2.txt")
    print("\nMin Conflicts")
    start = time.time()
    minConflicts_result, constraint_checks, visited_nodes = csp.min_conflicts(data)
    end = time.time()
    print(minConflicts_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)






    #######################
    ####### 7-w1-f4 #######
    #######################

    print("\n7-w1-f4\n")
    data = RLFA.Parsing("7-w1-f4.txt")
    print("FC")
    start = time.time()
    fc_result, constraint_checks, visited_nodes = csp.backtracking_search(data, select_unassigned_variable=csp.dom_wdeg, inference=csp.forward_checking)
    end = time.time()
    print(fc_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)
    
    data = RLFA.Parsing("7-w1-f4.txt")
    print("\nMAC")
    start = time.time()
    mac_result, constraint_checks, visited_nodes = csp.backtracking_search(data, select_unassigned_variable=csp.dom_wdeg, inference=csp.mac)
    end = time.time()
    print(mac_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)

    data = RLFA.Parsing("7-w1-f4.txt")
    print("\nMin Conflicts")
    start = time.time()
    minConflicts_result, constraint_checks, visited_nodes = csp.min_conflicts(data)
    end = time.time()
    print(minConflicts_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)







    #######################
    ####### 7-w1-f5 #######
    #######################

    print("\n7-w1-f5\n")
    data = RLFA.Parsing("7-w1-f5.txt")
    print("FC")
    start = time.time()
    fc_result, constraint_checks, visited_nodes = csp.backtracking_search(data, select_unassigned_variable=csp.dom_wdeg, inference=csp.forward_checking)
    end = time.time()
    print(fc_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)
    
    data = RLFA.Parsing("7-w1-f5.txt")
    print("\nMAC")
    start = time.time()
    mac_result, constraint_checks, visited_nodes = csp.backtracking_search(data, select_unassigned_variable=csp.dom_wdeg, inference=csp.mac)
    end = time.time()
    print(mac_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)

    data = RLFA.Parsing("7-w1-f5.txt")
    print("\nMin Conflicts")
    start = time.time()
    minConflicts_result, constraint_checks, visited_nodes = csp.min_conflicts(data)
    end = time.time()
    print(minConflicts_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)






    ######################
    ######## 8-f10 #######
    ######################

    print("\n8-f10\n")
    data = RLFA.Parsing("8-f10.txt")
    print("FC")
    start = time.time()
    fc_result, constraint_checks, visited_nodes = csp.backtracking_search(data, select_unassigned_variable=csp.dom_wdeg, inference=csp.forward_checking)
    end = time.time()
    print(fc_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)
    
    data = RLFA.Parsing("8-f10.txt")
    print("\nMAC")
    start = time.time()
    mac_result, constraint_checks, visited_nodes = csp.backtracking_search(data, select_unassigned_variable=csp.dom_wdeg, inference=csp.mac)
    end = time.time()
    print(mac_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)

    data = RLFA.Parsing("8-f10.txt")
    print("\nMin Conflicts")
    start = time.time()
    minConflicts_result, constraint_checks, visited_nodes = csp.min_conflicts(data)
    end = time.time()
    print(minConflicts_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)





    ######################
    ######## 8-f11 #######
    ######################

    print("\n8-f11\n")
    data = RLFA.Parsing("8-f11.txt")
    print("FC")
    start = time.time()
    fc_result, constraint_checks, visited_nodes = csp.backtracking_search(data, select_unassigned_variable=csp.dom_wdeg, inference=csp.forward_checking)
    end = time.time()
    print(fc_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)
    
    data = RLFA.Parsing("8-f11.txt")
    print("\nMAC")
    start = time.time()
    mac_result, constraint_checks, visited_nodes = csp.backtracking_search(data, select_unassigned_variable=csp.dom_wdeg, inference=csp.mac)
    end = time.time()
    print(mac_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)

    data = RLFA.Parsing("8-f11.txt")
    print("\nMin Conflicts")
    start = time.time()
    minConflicts_result, constraint_checks, visited_nodes = csp.min_conflicts(data)
    end = time.time()
    print(minConflicts_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)





    ######################
    ######### 11 #########
    ######################

    print("\n11\n")
    data = RLFA.Parsing("11.txt")
    print("FC")
    start = time.time()
    fc_result, constraint_checks, visited_nodes = csp.backtracking_search(data, select_unassigned_variable=csp.dom_wdeg, inference=csp.forward_checking)
    end = time.time()
    print(fc_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)
    
    data = RLFA.Parsing("11.txt")
    print("\nMAC")
    start = time.time()
    mac_result, constraint_checks, visited_nodes = csp.backtracking_search(data, select_unassigned_variable=csp.dom_wdeg, inference=csp.mac)
    end = time.time()
    print(mac_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)

    data = RLFA.Parsing("11.txt")
    print("\nMin Conflicts")
    start = time.time()
    minConflicts_result, constraint_checks, visited_nodes = csp.min_conflicts(data)
    end = time.time()
    print(minConflicts_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)






    #######################
    ######## 14-f27 #######
    #######################

    print("\n14-f27\n")
    data = RLFA.Parsing("14-f27.txt")
    print("FC")
    start = time.time()
    fc_result, constraint_checks, visited_nodes = csp.backtracking_search(data, select_unassigned_variable=csp.dom_wdeg, inference=csp.forward_checking)
    end = time.time()
    print(fc_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)
    
    data = RLFA.Parsing("14-f27.txt")
    print("\nMAC")
    start = time.time()
    mac_result, constraint_checks, visited_nodes = csp.backtracking_search(data, select_unassigned_variable=csp.dom_wdeg, inference=csp.mac)
    end = time.time()
    print(mac_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)

    data = RLFA.Parsing("14-f27.txt")
    print("\nMin Conflicts")
    start = time.time()
    minConflicts_result, constraint_checks, visited_nodes = csp.min_conflicts(data)
    end = time.time()
    print(minConflicts_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)





    #######################
    ######## 14-f28 #######
    #######################

    print("\n14-f28\n")
    data = RLFA.Parsing("14-f28.txt")
    print("FC")
    start = time.time()
    fc_result, constraint_checks, visited_nodes = csp.backtracking_search(data, select_unassigned_variable=csp.dom_wdeg, inference=csp.forward_checking)
    end = time.time()
    print(fc_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)
    
    data = RLFA.Parsing("14-f28.txt")
    print("\nMAC")
    start = time.time()
    mac_result, constraint_checks, visited_nodes = csp.backtracking_search(data, select_unassigned_variable=csp.dom_wdeg, inference=csp.mac)
    end = time.time()
    print(mac_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)

    data = RLFA.Parsing("14-f28.txt")
    print("\nMin Conflicts")
    start = time.time()
    minConflicts_result, constraint_checks, visited_nodes = csp.min_conflicts(data)
    end = time.time()
    print(minConflicts_result)
    print("Time elapsed: %.5f" % (end - start), "seconds")
    print("Constraint checks:", constraint_checks)
    print("Visited nodes:", visited_nodes)
