Setting: Package dependency graph

Goal: Resolve package dependencies

- component update, installation, removal

```python

struct Package:
    id
    version
    dependencies: set[Package]
# alt:
# packages can be represented as a relation set[tuple[id, version]] -> dependencies

repository: set[Package]

def flatten_dependencies(pkg: Package) -> set[Package]:
    return flatten(map(flatten_dependencies, pkg.dependencies))

def dependencies_conflict(pkg1: Package, pkg2: Package) -> bool:
    pkg1_dep = flatten_dependencies(pkg1)
    pkg2_dep = flatten_dependencies(pkg2)

    package_conflicts = filter(lambda p1,p2: p1.id == p2.id and p1.version != p2.version, times(pkg1_dep, pkg2_dep))
    if not package_conflicts:
        return True
    return False
```
