#define CATCH_CONFIG_RUNNER  // Tells Catch to not provide its own main()
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_session.hpp>

// Example function to be tested
int add(int a, int b) {
    return a + b;
}

// Test case
TEST_CASE("Addition works", "[math]") {
    REQUIRE(add(4, 5) == 9);
}

// Your own main function
int main(int argc, char* argv[]) {
    // Global setup can be done here

    // Run the tests and store the result
    int result = Catch::Session().run(argc, argv);

    // Global teardown can be done here

    return result;
}
