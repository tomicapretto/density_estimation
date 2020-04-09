import pkg_resources

REQUIRED_PKGS = {"numpy", "scipy"}

ALL_PKGS = {REQUIRED_PKGS.key for REQUIRED_PKGS in pkg_resources.working_set}
MISSING_PKGS = REQUIRED_PKGS - ALL_PKGS
AVAILABLE_PKGS = REQUIRED_PKGS - MISSING_PKGS

msg_list = ["Packages in use:"]

if AVAILABLE_PKGS != set():
  for PKG in AVAILABLE_PKGS:
    msg_list.append("{} v. {}".format(PKG, pkg_resources.get_distribution(PKG).version))

if MISSING_PKGS != set():
  for PKG in MISSING_PKGS:
    msg_list.append("{} NOT FOUND".format(PKG))

msg = "\n".join(msg_list)
print(msg)

assert REQUIRED_PKGS == AVAILABLE_PKGS, "Not all required packages have been found."

