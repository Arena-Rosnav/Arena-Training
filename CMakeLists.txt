cmake_minimum_required(VERSION 3.8)
project(arena_training)

# Create symlinks for rosnav_rl submodule packages to be discoverable by colcon
# This allows the git submodule packages to be built as part of the workspace
set(ROSNAV_RL_SUBMODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/rosnav_rl")
set(WORKSPACE_DEPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../deps")

# Create workspace deps directory if it doesn't exist
file(MAKE_DIRECTORY "${WORKSPACE_DEPS_DIR}")

# Symlink rosnav_rl package
set(ROSNAV_RL_SYMLINK "${WORKSPACE_DEPS_DIR}/rosnav_rl")
if(EXISTS "${ROSNAV_RL_SUBMODULE_DIR}/rosnav_rl" AND NOT EXISTS "${ROSNAV_RL_SYMLINK}")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
      "${ROSNAV_RL_SUBMODULE_DIR}/rosnav_rl"
      "${ROSNAV_RL_SYMLINK}"
    RESULT_VARIABLE SYMLINK_RESULT
  )
  if(SYMLINK_RESULT EQUAL 0)
    message(STATUS "Created symlink: ${ROSNAV_RL_SYMLINK} -> ${ROSNAV_RL_SUBMODULE_DIR}/rosnav_rl")
  else()
    message(WARNING "Failed to create symlink for rosnav_rl")
  endif()
endif()

# Symlink rosnav_rl_msgs package
set(ROSNAV_RL_MSGS_SYMLINK "${WORKSPACE_DEPS_DIR}/rosnav_rl_msgs")
if(EXISTS "${ROSNAV_RL_SUBMODULE_DIR}/rosnav_rl_msgs" AND NOT EXISTS "${ROSNAV_RL_MSGS_SYMLINK}")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
      "${ROSNAV_RL_SUBMODULE_DIR}/rosnav_rl_msgs"
      "${ROSNAV_RL_MSGS_SYMLINK}"
    RESULT_VARIABLE SYMLINK_RESULT
  )
  if(SYMLINK_RESULT EQUAL 0)
    message(STATUS "Created symlink: ${ROSNAV_RL_MSGS_SYMLINK} -> ${ROSNAV_RL_SUBMODULE_DIR}/rosnav_rl_msgs")
  else()
    message(WARNING "Failed to create symlink for rosnav_rl_msgs")
  endif()
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Python package installation
ament_python_install_package(${PROJECT_NAME} SCRIPTS_DESTINATION lib/${PROJECT_NAME})

# Install Python scripts
install(PROGRAMS
  scripts/train_agent.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install config files to package share directory
install(DIRECTORY configs
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
