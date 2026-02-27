from RubiksCubeSolver import RubiksCubeSolver
from convert_solution_to_moves import convert_solution_to_moves
from ArduinoCommunication import ArduinoCommunication     

solver = RubiksCubeSolver(5)
solution = solver.run()
converted_solution = convert_solution_to_moves(solution)
arduino = ArduinoCommunication()
arduino.send_to_arduino(converted_solution)