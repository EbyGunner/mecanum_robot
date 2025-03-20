cmake_minimum_required(VERSION 3.8)
project(mecanum_robot_simulation)


# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ros_gz_sim)
find_package(xacro REQUIRED)
find_package(robot_state_publisher REQUIRED)

# add_executable(test_gz_vendor main.cc)
# target_link_libraries(test_gz_vendor PUBLIC gz-math::gz-math gz-utils::gz-utils)

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(DIRECTORY src/launch/ 
  DESTINATION share/${PROJECT_NAME}/src/launch
  )

install(DIRECTORY src/description/urdf/
  DESTINATION share/${PROJECT_NAME}/src/description/urdf
  )

install(DIRECTORY src/world/
  DESTINATION share/${PROJECT_NAME}/src/world
  )  

install(DIRECTORY src/config/
  DESTINATION share/${PROJECT_NAME}/src/config
  )  

install(DIRECTORY src/rviz/
  DESTINATION share/${PROJECT_NAME}/src/rviz)

ament_package()
