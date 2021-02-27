import csp_dom_wdeg as csp


# A class to parse the data of the RLFA CSP
class Parsing(csp.CSP):
    def __init__(self, file_name):
        # Read the variables and domain indexes of the CSP
        variables_domain = []
        f = open("var"+file_name, "r")
        # Ignore the first line which is the amount of the records that follow
        first_line = f.readline()
        for line in f:
            line = line.strip()
            variables_domain.append(line.split(" "))
        f.close()


        # Append the variables to a list
        self.variables = []
        for item in variables_domain:
            self.variables.append(int(item[0]))


        # Read the domain indexes and their values
        domains_values = []
        f = open("dom" + file_name, "r")
        # Ignore the first line which is the amount of the records that follow
        first_line = f.readline()
        for line in f:
            line = line.strip()
            domains_values.append(line.split(" "))


        # A dictionary of {var:[possible_value, ...]} entries.
        self.variables_lists_of_values = {}
        # For each domain of a variable, search its values in the domains_value list
        for variable_domain in variables_domain:
            list_of_values = []
            for item in domains_values:
                if variable_domain[1] == item[0]:
                    # Ignore the first number which is the domain index and the second number
                    # which is the amount of records that follow
                    for value in item[2:]:
                        list_of_values.append(int(value))
                    self.variables_lists_of_values.update({int(variable_domain[0]): list_of_values})
                    break
        f.close()


        # We need a list that contains lists of the constraints splitted
        # and a dictionary with the constraint as key and the weight as value for the dom/wdeg heuristic
        self.constraints_splitted = []
        self.weights = {}
        # Read the constraints of the CSP
        f = open("ctr" + file_name, "r")
        # Ignore the first line which is the amount of the records that follow
        first_line = f.readline()
        for line in f:
            line = line.strip()
            self.weights.update({line: 1})
            self.constraints_splitted.append(line.split(" "))
        f.close()


        # A dictionary of {var:[var,...]} that for each variable lists
        # the other variables that participate in constraints
        self.neighbors = {}
        for variable in self.variables:
            list_of_neighbors = []
            for constraint in self.constraints_splitted:
                # Search if variable participates in whichever of the two
                # parts of the constraint
                if int(constraint[0]) == variable:
                    list_of_neighbors.append(int(constraint[1]))
                if int(constraint[1]) == variable:
                    list_of_neighbors.append(int(constraint[0]))
            self.neighbors.update({variable: list_of_neighbors})


        # A dictionary which has a tuple of the two variables (A, B) as key and
        # a list of the constraints splitted, as value
        # This is useful for the constraints function, in order to search in O(1)
        # all the constraints between two variables
        self.tuples_constraints = {}
        for constraint in self.constraints_splitted:
            list_of_constraints = self.tuples_constraints.get((int(constraint[0]), int(constraint[1])))
            if list_of_constraints:
                list_of_constraints.append(constraint)
            else:
                list_of_constraints = []
                list_of_constraints.append(constraint)
                self.tuples_constraints.update({(int(constraint[0]), int(constraint[1])): list_of_constraints})


        # We don't need these anymore
        variables_domain.clear()
        domains_values.clear()

        # Initialize the CSP
        csp.CSP.__init__(self, self.variables, self.variables_lists_of_values, self.neighbors, self.check_constraint, self.weights, self.tuples_constraints)


    # A function f(A, a, B, b) that returns true if neighbors
    # A, B satisfy the constraint when they have values A=a, B=b
    # If false, also return the constraint not satisfied (for usage in dom/wdeg heuristic)
    def check_constraint(self, A, a, B, b):
        # A counter for how many constraints have been checked
        self.constraints_count += 1
        # Get a list of the constraints (splitted) between A and B
        list_of_constraints = self.tuples_constraints.get((A, B))
        if list_of_constraints:
            for constraint in list_of_constraints:
                if constraint[2] == ">":
                    if abs(a-b) > int(constraint[3]):
                        continue
                    else:
                        full_constraint = " ".join(constraint)
                        return False, full_constraint
                elif constraint[2] == "=":
                    if abs(a-b) == int(constraint[3]):
                        continue
                    else:
                        full_constraint = " ".join(constraint)
                        return False, full_constraint
        # Get a list of the constraints (splitted) between B and A
        list_of_constraints = self.tuples_constraints.get((B, A))
        if list_of_constraints:
            for constraint in list_of_constraints:
                if constraint[2] == ">":
                    if abs(a-b) > int(constraint[3]):
                        continue
                    else:
                        full_constraint = " ".join(constraint)
                        return False, full_constraint
                elif constraint[2] == "=":
                    if abs(a-b) == int(constraint[3]):
                        continue
                    else:
                        full_constraint = " ".join(constraint)
                        return False, full_constraint
        return True, 0
