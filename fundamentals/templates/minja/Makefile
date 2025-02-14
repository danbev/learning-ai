CXXFLAGS := -g -Wall -std=c++20 -Isrc

example: src/example.cpp
	@$(CXX) $(CXXFLAGS) -o example src/example.cpp 

run: example
	@./example

debug: example
	gdb --args ./example

ctags:
	ctags -R .

update-minja:
	wget -P src https://raw.githubusercontent.com/google/minja/main/include/minja/minja.hpp
	wget -P src https://raw.githubusercontent.com/google/minja/main/include/minja/chat-template.hpp

update-json:
	wget -P src https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp

clean:
	${RM} example

