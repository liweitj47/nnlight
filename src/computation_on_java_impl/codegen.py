class CodeGenerator:

    def __init__(self):
        self.__code = ""
        self.indent = 0

    def append(self, line=""):
        for _ in range(self.indent):
            self.__code += "\t"
        self.__code += line
        self.__code += "\n"

    def code(self):
        return self.__code

    def escape(self, name):
        return name.replace(".", "$")

    def begin_class(self, name, scope="public"):
        line = "%s class %s {" % (scope, name)
        self.append(line)
        self.indent += 1

    def field(self, typ, name, scope=None, val=None):
        line = (scope + " ") if scope else ""
        line += typ + " " + name
        if val is not None:
            line += " = " + str(val)
        line += ";"
        self.append(line)

    def begin_function(self, return_type, name, params=None, scope="public"):
        line = scope + " " if scope else ""
        line += return_type + " " if len(return_type) > 0 else ""
        line += name + "("
        if params:
            params = [t + " " + n for t, n in params]
            line += str.join(", ", params)
        line += ") {"
        self.append(line)
        self.indent += 1

    def end(self):
        self.indent -= 1
        self.append("}")

    def assignment(self, l, r, operator="="):
        self.append(l + " " + operator + " " + str(r) + ";")

    def call(self, funcname, params, target=None):
        line = ""
        if target:
            line += target + " = "
        line += funcname + "(" + str.join(", ", params) + ");"
        self.append(line)

    def new_array(self, target, basetype, dims, scope=None):
        line = scope + " " if scope else ""
        line += basetype + " " + str.join("", ["[]" for _ in dims]) + " "
        line += target
        line += " = new " + basetype
        line += str.join("", ["[" + ("" if d is None else str(d)) + "]" for d in dims])
        line += ";"
        self.append(line)

    def new(self, target, classname, params=None, scope=None):
        line = scope + " " if scope else ""
        line += classname + " " + target + " = new " + classname
        if params:
            line += "(" + str.join(", ", params) + ");"
        else:
            line += "();"
        self.append(line)

    def returnn(self, target=""):
        self.append("return " + target + ";")

    def package(self, package):
        self.append("package " + package + ";")

    def importt(self, package):
        self.append("import " + package + ";")

    def begin_for(self, exp):
        self.append("for(" + exp + ") {")
        self.indent += 1

    def begin_if(self, cond):
        self.append("if(" + cond + ") {")
        self.indent += 1

    def begin_else(self):
        self.append("else {")
        self.indent += 1
