# benes_tools

this repo holds some helper functions to have them easily accessible accross machines (and version controlled).


## Unit tests

### test_roations

test roundtrip from n quaternions (uniformly distributed over SO(3), generated with super-fibonacci) to rotation matrices, to euler angles and back to quaternions. 

run all rotation tests:

```uv run --group test pytest -v ```

and/or use flag ```-s``` to show print and stderr outputs
