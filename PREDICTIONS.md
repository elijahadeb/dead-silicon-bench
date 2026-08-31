# gate -1, here are my predictions, before i write any code.

## what i think each arm will do

scalar:    11.2 gb/s, because since my cpu processing speed is 2.8 billion operations per seconds and in a 32 bits memory - 100 million numbers will take up 0.4GB, and since scalar instructions read data one by one - we mulitply 2.8 (billion operations) * 4 (bytes) = 11.2 gb/s.
avx2:      34 gb/s flatlined, because the avx2 gives an extra wide registers of 256 bit (32 bytes). we can predict from here. this means it can read 4 extra numbers at once. so it becomes: 256/32 = 8
then 11.2 gb/s * 8 = 89.6 gb/s (theoretical compute limit). for the full speed. 
threaded:  34 gb/s, because threaded avx2 involves running avx2 across the 4 cpu cores, meaning it's processing 256 bits of data in 4 separate places at the exact time. so the math becomes - 256 * 4 = 1024, then we figure out how many number fits into the bit space 1024. which is 1024/32 = 32. so the speed becomes... 11.2 * 32 = 358 gb/s (this is the theoretical compute limit). but this won't be the speed because threaded on this machine is memory bound - due to the laptops dual channel DDR4 memory bus limit... it caps total bandwidth at roughly 34 gb/s.

## the shape

[x] avx2 is a large jump over scalar
[x] threaded is a small jump over avx2, then flat
[x] flattening happens at 1st threads (or core)
[x] none of the three exceeds 38.0 gb/s, because that is the wall
