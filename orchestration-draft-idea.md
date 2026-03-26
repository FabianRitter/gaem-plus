
Why we need so many agents:
Research unpredictability makes AI agents particularly well-suited for research tasks. Research demands the flexibility to pivot or explore tangential connections as the investigation unfolds. The model must operate autonomously for many turns, making decisions about which directions to pursue based on intermediate findings. A linear, one-shot pipeline cannot handle these tasks.


I need to orchestrate sub agents pipelines so I they can study this idea further and we can come up with a final decision before we start moving into the implementation phase. We will also need to create an orchestration once we move into the implementation phase.
Let me know what do you think about this:

2 lead agent orchestrator like a project manager/researchers. They will discuss between each other to go to final solution. 1 of them will read all my notes for the idea plus the suggestions of the agents and the other acts the critic that will listen and will try to challenge the idea or take it to a further step. Once they agree in this, they will set up the final plan for experimenting.

2 search subagents that search papers and summarises them in the excel we have.
2 sub agents that look at the added papers in the excel and discuss with each other how to come up with a research idea based on this readings as well as all my notes on generalised unified model merging of speech audio and music that I have in this folders. A key step I think is to move directly to LLM application such as the SALMONN example instead of analysing SSL models as I need to catch up with LLM asap.

1 sub agent that study datasets and suggests a data pipeline based on available corpora and provides a codebase to download this data to a certain path (we do not download the data here)
1 sub agent that will get the final plan of the 2 lead agents and it will find a base codebase to start this idea and from this start of codebase it will design a codebase plan to make changes to get our idea going. Our actual implementation of everything it will be done later once we move to the GPU cluster but we need a full documentation to achieve this and we may need to create a new orchestration for this we may want to make a CLAUDE.md for this step that states a single agent to first download base repository and the datasets and then we can add the orchestration. 
1 teacher agent that reads the code and the research idea and write reports on how the idea is being implemented so that I can learn about the techniques used in more detail.

Examples to achieve this multi agent orchestration:
https://github.com/bobmatnyc/claude-multiagent-pm
https://www.anthropic.com/engineering/multi-agent-research-system

Feel free to use subagents to creates this idea of multi agents for my research. And suggest how to move from the idea we are doing here to actually doing the things in claude code to start writing  the code, if you believe this step is better to be done with claude code from the beginning and to integrate this research with the implementation then please stop this task asap and suggest me which files to put in my cluster from here and how to get things started. I suspect claude code will not be able to read the slides but we can focus on the excel and all the .md files.

Additional important information for when it comes to do the building of the code and for when we will need to debug the code:

Datasets should go in : /data/projects/12004380/datasets
Cluster uses qsub and enroot, cannot do debugging without entering a node session. You may find some running jobs that I may be running and then do
export PBS_JOBID=(jobidname)
Ssh jobid-node, example:

export PBS_JOBID=149915.pbs111
qstat -f 149915.pbs111 -> find node name in here or another trick you may know.
An example that shows the enroot that we have in nscc cluster:
/data/projects/12004380/fabian/task-arithmetic-speech-audio/s3prl/run_downstream_enroot_2026.sh

You may want to use the same enroot and install packages as needed on top of this one. It may be the easier way but feel free to download your own docker image and build on top of it or ask me to do something related to this.

PS: we are already in the directory and in the cluster where I want to set up this idea so please plan how we can proceed with all of these tasks step by step. 

