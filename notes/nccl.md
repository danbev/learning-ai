## NVIDIA Collective Communications Library ("Nickel")
This component (software component?) handles the controller part of communications
between GPUs, called collectives, so instead of sending data point to  point 
between GPUs it enables more efficient communication patterns like broadcast,
all-reduce.

* Broadcast: One GPU sends data to all other GPUs.
* Reduce: All GPUs sends a number to one GPU, which sums them up.
* All-reduce: All GPUs have a number and they call sum them up and everyone gets
              the final total.
