class Simulation:
    def __init__(self, speed, dt, finish_process):
        """
        TAKES CARE OF SIMULATION OF EXPERIMENTS
        :param speed (float): speed of each drone m/s
        :param dt (float): time period of system s
        :param assignments_each_iter (bool): determines whether or not we reevaluate assignments at each increment
        """
        self.speed = speed
        self.dt = dt
        self.time_elapsed_list = [0]
        self.number_of_checkpoints_list = []
        self.finish_process = finish_process

        self.store_directions = None

    def simulate(self, world_object, controller_object, stop_after=None):
        """
        TAKES CARE OF ENTIRE SIMULATION PROCESS
        :param world_object: contains all world information
        :param controller_object: contains all controller functions
        """
        self.number_of_checkpoints_list.append(world_object.checkpoints_coord.shape[0])
        while (len(world_object.remaining_checkpoints) > 0 and self.finish_process) or (
                len(world_object.remaining_checkpoints) > world_object.n_agents):  ############
            if len(self.number_of_checkpoints_list) == 1 or not self.number_of_checkpoints_list[-1] == \
                                                                self.number_of_checkpoints_list[-2]:
                reevaluate_assignments = True
            else:
                reevaluate_assignments = False
            self.world_step(world_object, controller_object, reevaluate_assignments)
            self.number_of_checkpoints_list.append(len(world_object.remaining_checkpoints))
            self.time_elapsed_list.append(self.time_elapsed_list[-1] + self.dt)
            print("%.2f" % self.time_elapsed_list[-1], self.number_of_checkpoints_list[-1])
            if stop_after < self.time_elapsed_list[-1]:
                break

    def world_step(self, world_object, controller_object, reevaluate_assignments):
        """
        TAKES CARE OF A WORLD STEP
        :param world_object: contains all world information
        :param controller_object: contains all controller functions
        """

        self.store_directions = controller_object.get_directions(world_object)
        agent_directions = self.store_directions

        world_object.agents_coord = world_object.agents_coord + self.speed * self.dt * agent_directions
        purge_list = world_object.purge_checkpoints(assignment_group=controller_object.assignment_group,
                                                    reached_radius=self.speed * self.dt)
