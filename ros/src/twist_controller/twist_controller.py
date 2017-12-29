from yaw_controller import YawController
from pid import PID
import numpy as np
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, wheel_base, steer_ratio,
                 min_speed, max_lat_accel, max_steer_angle, vehicle_mass,
                 fuel_capacity, decel_limit, wheel_radius,friction_coef=0.5):
        # TODO: Implement

        total_mass = vehicle_mass + fuel_capacity * GAS_DENSITY

        self.torque = total_mass * wheel_radius * np.abs(decel_limit) * friction_coef

        self.yaw_controller = YawController(wheel_base, steer_ratio,
                                            min_speed, max_lat_accel, max_steer_angle)

        self.linear_velocity_controller=PID(0.1,0,0)

    def control(self, linear_velocity, angular_velocity, current_velocity):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer

        steer = self.yaw_controller.get_steering(linear_velocity,
                                                 angular_velocity, current_velocity)

        sample_time=0.02
        velocity_error=linear_velocity-current_velocity

        pid_val=self.linear_velocity_controller.step(velocity_error,sample_time)
        rospy.loginfo("pid value: %f",pid_val)

        if pid_val<0:
            throttle=0
            brake=self.torque*0.2
        else:
            throttle=0.2
            brake=0

        return throttle, brake, steer
