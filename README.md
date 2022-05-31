# MRWP_Project
Students: Emiel Robben, Leah Dollo, Tim Döring, YOOOOP

Idea: Lets make this our todo-list and slowly transition it into a workload distribution record

Final report: https://www.overleaf.com/9652717651bmnjydqwzqbx

UPDATES:

1) We decided to focus on Zurich (instead of all of Switzerland)
2) How did gouvernment restrictions in response to the COVID-19 pandemic affect first year student's ability to find new friends? An ABM model combined with
      a social network, simulating the case of Zurich based on real data.
      
      
 Project Structure (as proposed by Tim):
 
      - Initialize the map of Zurich more or less realistically (libraries for students, occasions for recreational time spending (bars, cinemas, restaurants, etc), university campus, ...)
      - Find a realistic ratio between first year students and other students/young adults that are relevant for a social network
      - Initialize a social network between the present young adults (those that lived in Zurich before COVID)
      - Add new agents without social ties to the spacial model (aka our fist years that don't know anyone and just moved there)
      - Let the model run without simulated covid restrictions --> students should be able to find friends in a more or less quick way
      - Find covid-measures implemented by the swiss government and translate them to parameter changes in our model
      - Run the simulation again and see how the ability to find new friends differs in comparison to the fist run

MOST URGENT:

1) Check present sources in more detail, to see what useful information they contain
2) Based on that, decide what to cut down on & What do focus on. Plan of action for our model?
3) Based on that, find even more sources

FIRST STEPS - CODING

1) Push relevant datasets to git (create "data" folder first)
2) Read CSVs into suitable datastructure (nested list or dict), filter out or correct cells if necessary

FIRST STEPS - REPORT

1) Look up source 9 (pwp) for current limitations on combining ABMs with Social Networks
