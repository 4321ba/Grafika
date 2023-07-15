#include <map>
#include "framework.h"


void prf(float f, const char* ch = "") {
	printf("%s: %f\n", ch, f);
}
void prv(vec3 v, const char* ch = "") {
	printf("%s: (%f, %f, %f)\n", ch, v.x, v.y, v.z);
}

int realWindowWidth = windowWidth;
int realWindowHeight = windowHeight;

//--------------------------
struct vec3i {
//--------------------------
	int x, y, z;

	vec3i(int x0 = 0, int y0 = 0, int z0 = 0) { x = x0; y = y0; z = z0; }
	vec3i(vec3 v) { x = roundf(v.x); y = roundf(v.y); z = roundf(v.z); }
	operator vec3() const {return vec3(x, y, z);}
	bool operator==(const vec3i& v) const { return x == v.x && y == v.y && z == v.z; }
	bool operator<(const vec3i& v) const {
		if (x != v.x)
			return x < v.x;
		if (y != v.y)
			return y < v.y;
		return z < v.z;
	}

	vec3i operator*(int a) const { return vec3i(x * a, y * a, z * a); }
	vec3i operator+(const vec3i& v) const { return vec3i(x + v.x, y + v.y, z + v.z); }
	vec3i operator-(const vec3i& v) const { return vec3i(x - v.x, y - v.y, z - v.z); }
	vec3i operator*(const vec3i& v) const { return vec3i(x * v.x, y * v.y, z * v.z); }
	vec3i operator-()  const { return vec3i(-x, -y, -z); }
};

inline vec3i operator*(int a, const vec3i& v) { return vec3i(v.x * a, v.y * a, v.z * a); }


class Camera {
	vec3 wEye = vec3(0,4,2);
	vec3 wForward = vec3(0,0,-1); // y mindig 0
	float angleUp = -0.2;
	float asp = 1;
	
	const vec3 wVup = vec3(0,1,0);
	const float fov = 70 * 2* M_PI / 360.0;
	const float fp = 0.02;
	const float bp = 200;
public:
	vec3 getPos() { return wEye; }
	mat4 V() { // view matrix
		
		vec3 w = normalize(-wLookDir());
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(-wEye) * mat4(
			u.x, v.x, w.x, 0,
			u.y, v.y, w.y, 0,
			u.z, v.z, w.z, 0,
			0, 0, 0, 1);
	}
	mat4 P() { // projection matrix
		float sy = 1/tanf(fov/2);
		asp = realWindowWidth / (float)realWindowHeight;
		return mat4(
			sy/asp, 0, 0, 0,
			0, sy, 0, 0,
			0, 0, -(fp+bp)/(bp-fp), -1,
			0, 0, -2*fp*bp/(bp-fp), 0);
	}
	
	vec3 wLookDir() {
		vec4 wLookDir = vec4(wForward.x, wForward.y, wForward.z, 0) * RotationMatrix(angleUp, normalize(cross(wForward, wVup)));
		return vec3(wLookDir.x, wLookDir.y, wLookDir.z);
	}
	
	vec3 wVelocity;
	// dir:  -z = elore, +y = fel, +x = jobbra, tehat sajat koordrendszerben
	void move(vec3 dir, float delta) {
		static const float kozegell = 5;
		vec3 wAccel = -wForward * dir.z + wVup * dir.y + cross(wForward, wVup) * dir.x - wVelocity * kozegell;
		wVelocity = wVelocity + wAccel * delta;
		wEye = wEye + wVelocity * delta;
	}
	// dir: +x: jobbra, +y: fel
	void rotate(vec2 dir) {
		vec4 v = vec4(wForward.x, wForward.y, wForward.z, 0) * RotationMatrix(dir.x, wVup);
		wForward = normalize(vec3(v.x, v.y, v.z));
		static const float epsilon = 0.00001;
		if (angleUp + dir.y >= M_PI / 2)
			angleUp = M_PI / 2 - epsilon;
		else if (angleUp + dir.y <= -M_PI / 2)
			angleUp = -M_PI / 2 + epsilon;
		else
			angleUp += dir.y;
	}
};

Camera camera;
GPUProgram gpuProgram;
struct combinedTexture {
	Texture *bottom, *side, *top;
	void load(std::string bot, std::string sid, std::string to) {
		// https://gamedev.stackexchange.com/questions/19075/how-can-i-make-opengl-textures-scale-without-becoming-blurry
		bottom = new Texture(bot);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		side = new Texture(sid);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		top = new Texture(to);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	}
};
combinedTexture textures[10];

struct Hit {
    vec3i pos, normal = vec3(0,0,0);
	float t = -1; // < 0 ha ervenytelen
};

struct Block {
	vec3i pos;
	unsigned texidx;
	Block() {}
	Block (vec3i pos, unsigned texidx): pos(pos), texidx(texidx) {}
	void Draw() {
		mat4 M = ScaleMatrix(vec3(1,1,1)) *
				RotationMatrix(0, vec3(0,1,0)) *
				TranslateMatrix(pos);
		mat4 Minv = TranslateMatrix(-pos) *
				RotationMatrix(-0, vec3(0,1,0)) *
				ScaleMatrix(vec3(1/1,1/1,1/1));
		mat4 MVP = M * camera.V() * camera.P();
		gpuProgram.setUniform(M, "M");
		gpuProgram.setUniform(Minv, "Minv");
		gpuProgram.setUniform(MVP, "MVP");
		gpuProgram.setUniform(*textures[texidx].bottom, "textureUnitBottom", 0);
		gpuProgram.setUniform(*textures[texidx].side, "textureUnitSide", 1);
		gpuProgram.setUniform(*textures[texidx].top, "textureUnitTop", 2);
		glDrawArrays(GL_TRIANGLES, 0, 36);
	}
	vec3 divide(const vec3& v, const vec3& w) { return vec3(v.x / w.x, v.y / w.y, v.z / w.z); }
	vec3 min(const vec3& v, const vec3& w) { return vec3(std::min(v.x, w.x), std::min(v.y, w.y), std::min(v.z, w.z)); }
	vec3 max(const vec3& v, const vec3& w) { return vec3(std::max(v.x, w.x), std::max(v.y, w.y), std::max(v.z, w.z)); }
	// a hardveres vagosikos dologbol
	Hit intersect(vec3 start, vec3 dir) {
        dir = normalize(dir);
        vec3 end = start + 20.0 * dir;
        vec3i posmax = pos + vec3i(1, 1, 1);
        // nagyobb koordinataju sikokkal valo metszespontok
        vec3 tupper = divide((vec3)posmax - start, end - start);
		// nan or infinite: ha pont parhuzamos
		if (!std::isfinite(tupper.x)) tupper.x = INFINITY;
		if (!std::isfinite(tupper.y)) tupper.y = INFINITY;
		if (!std::isfinite(tupper.z)) tupper.z = INFINITY;
        // kisebbekkel
        vec3 tlower = divide((vec3)pos - start, end - start);
		if (!std::isfinite(tlower.x)) tlower.x = -INFINITY;
		if (!std::isfinite(tlower.y)) tlower.y = -INFINITY;
		if (!std::isfinite(tlower.z)) tlower.z = -INFINITY;
        // koordinatankent:
        vec3 tmin = min(tlower, tupper);
        vec3 tmax = max(tlower, tupper);
        float inmax = std::max(tmin.x, std::max(tmin.y, tmin.z));
        float outmin = std::min(tmax.x, std::min(tmax.y, tmax.z));
        Hit h;
        if (inmax > outmin || inmax < 0 || inmax > 1) // ha nem talaljuk el, vagy mogottunk van, vagy tul messze van
            return h;
		// hirtelen nem tudok jobbat arra, hogy hogyan kapjam meg, hogy a legkisebb t melyik oldalhoz tartozik:
		float ts[] = { tlower.x, tlower.y, tlower.z, tupper.x, tupper.y, tupper.z };
		vec3 normals[] = {
			vec3(-1, 0, 0),
			vec3( 0,-1, 0),
			vec3( 0, 0,-1),
			vec3( 1, 0, 0),
			vec3( 0, 1, 0),
			vec3( 0, 0, 1),
		};
		for (int i = 0; i < 6; ++i)
			if (inmax == ts[i]) {
				h.normal = normals[i];
				break;
			}
			
		//prf(inmax, "inmax");
		//prf(outmin, "outmin");
        h.pos = pos;
		h.t = inmax;
        return h;
    }
};

std::map<vec3i, Block> blocks;
void placeBlock(vec3i pos, unsigned type) {
	blocks[pos] = Block(pos, type);
}
void deleteBlock(vec3i pos) {
	blocks.erase(pos);
}



// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP, M, Minv;			// uniform variable, the Model-View-Projection transformation matrix
	uniform vec3 wEye;
	
	layout(location = 0) in vec3 vp;	// Varying input: vp = vertex position is expected in attrib array 0
	layout(location = 1) in vec3 normal;
	
	out vec4 color;
	out vec2 texcoord;
	out float side; // -1 ha bottom, 0 ha side es 1 ha top
	
	void main() {
		gl_Position = vec4(vp, 1) * MVP;		// transform vp from modeling space to normalized device space
		vec4 wPos = vec4(vp, 1) * M;
		vec4 wNormal = Minv * vec4(normal, 0);
		float c = 0.4 * (1 + dot(wNormal.xyz, normalize(wEye - wPos.xyz)));
		color = vec4(c,c,c,1);
		
		if (normal.y < -0.5) {
			side = -1;
			texcoord = vp.xz;
		} else if (normal.y > 0.5) {
			side = 1;
			texcoord = vp.xz;
		} else {
			side = 0;
			if (normal.x < -0.5 || normal.x > 0.5)
				texcoord = vp.zy;
			else
				texcoord = vp.xy;
		}
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform sampler2D textureUnitBottom;
	uniform sampler2D textureUnitSide;
	uniform sampler2D textureUnitTop;
	
	in float side;
	in  vec2 texcoord;			// interpolated texture coordinates
	in vec4 color;		// uniform variable, the color of the primitive
	
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		vec4 texcolor;
		if (side < -0.5)
			texcolor = /*vec4(0,texcoord.x,texcoord.y,1);//*/texture(textureUnitBottom, texcoord);//bottom
		else if (side > 0.5)
			texcolor = /*vec4(texcoord.x,texcoord.y,0,1);//*/texture(textureUnitTop, texcoord);//top
		else 
			texcolor = /*vec4(1,texcoord.x,texcoord.y,1);//*/texture(textureUnitSide, texcoord);//side
		outColor = texcolor * color;	// computed color is the color of the primitive
	}
)";

bool captureMouse = false;
// https://gamedev.stackexchange.com/questions/10100/trapping-mouse-inside-window-in-opengl-with-glut
void onMouseMotion(int x, int y) {
	if (!captureMouse)
		return;
    int centerX = realWindowWidth / 2;
    int centerY = realWindowHeight / 2;
    if(x == centerX && y == centerY)
		return;

	static vec2 prevrot[2];
	vec2 rotation = vec2(centerX - x, centerY - y) * 0.0007; // mouse sensitivity
	//prv(rotation, "rot");
	camera.rotate(rotation * 0.6 + prevrot[0] * 0.3 + prevrot[1] * 0.1);
	prevrot[1] = prevrot[0];
	prevrot[0] = rotation;

	glutPostRedisplay();
	glutWarpPointer(centerX, centerY);
}

// https://stackoverflow.com/questions/3867441/resizing-glut-window
void onWindowReshape(int newWidth, int newHeight) {
	glViewport(0, 0, newWidth, newHeight);
	realWindowWidth = newWidth;
	realWindowHeight = newHeight;
	glutPostRedisplay();
	//prv(vec3(realWindowWidth, realWindowHeight, 0),"hv");
}

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
    //glEnable(GL_CULL_FACE);

	// kocka feltoltese a gpu-ra
	unsigned int vao;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	unsigned int vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	float vertices[] = { 0, 0, 0, 0, 0, 1, 0, 1, 0,        0, 1, 1, 0, 0, 1, 0, 1, 0,
						 1, 0, 0, 1, 0, 1, 1, 1, 0,        1, 1, 1, 1, 0, 1, 1, 1, 0,
						 0, 0, 0, 0, 0, 1, 1, 0, 0,        1, 0, 1, 0, 0, 1, 1, 0, 0,
						 0, 1, 0, 0, 1, 1, 1, 1, 0,        1, 1, 1, 0, 1, 1, 1, 1, 0,
						 0, 0, 0, 1, 0, 0, 0, 1, 0,        1, 1, 0, 1, 0, 0, 0, 1, 0,
						 0, 0, 1, 1, 0, 1, 0, 1, 1,        1, 1, 1, 1, 0, 1, 0, 1, 1,
	/*};
	float normals[] ={*/-1, 0, 0,-1, 0, 0,-1, 0, 0,       -1, 0, 0,-1, 0, 0,-1, 0, 0,
						 1, 0, 0, 1, 0, 0, 1, 0, 0,        1, 0, 0, 1, 0, 0, 1, 0, 0,
						 0,-1, 0, 0,-1, 0, 0,-1, 0,        0,-1, 0, 0,-1, 0, 0,-1, 0,
						 0, 1, 0, 0, 1, 0, 0, 1, 0,        0, 1, 0, 0, 1, 0, 0, 1, 0,
						 0, 0,-1, 0, 0,-1, 0, 0,-1,        0, 0,-1, 0, 0,-1, 0, 0,-1,
						 0, 0, 1, 0, 0, 1, 0, 0, 1,        0, 0, 1, 0, 0, 1, 0, 0, 1,
	};
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)(sizeof(vertices) / 2));
	
	textures[0].load("dirt.bmp", "crafting_table_front.bmp", "crafting_table_top.bmp");
	textures[1].load("dirt.bmp", "dirt.bmp", "dirt.bmp");
	textures[2].load("dirt.bmp", "grass_block_side.bmp", "grass_block_top.bmp");
	textures[3].load("acacia_planks.bmp", "acacia_planks.bmp", "acacia_planks.bmp");
	textures[4].load("acacia_planks.bmp", "acacia_planks.bmp", "acacia_planks.bmp");
	textures[5].load("acacia_planks.bmp", "acacia_planks.bmp", "acacia_planks.bmp");
	textures[6].load("acacia_planks.bmp", "acacia_planks.bmp", "acacia_planks.bmp");
	textures[7].load("acacia_planks.bmp", "acacia_planks.bmp", "acacia_planks.bmp");
	textures[8].load("acacia_planks.bmp", "acacia_planks.bmp", "acacia_planks.bmp");
	textures[9].load("acacia_planks.bmp", "acacia_planks.bmp", "acacia_planks.bmp");
	
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
	
	placeBlock(vec3i(0,1,0), 0);
	for (int i = -4; i < 5; ++i)
		for (int j = -4; j < 5; ++j) {
			if (i >= 0 || j >= 0) {
				placeBlock(vec3i(i,0,j), 2);
				placeBlock(vec3i(i,-1,j), 1);
			} else
				placeBlock(vec3i(i,-1,j), 2);
				
		}
	for (int x = 1; x < 4; ++x)
		for (int y = 0; y < 3; ++y)
			for (int z = 1; z < 4; ++z)
				placeBlock(vec3i(x,y,z), 3);
	
	glutPassiveMotionFunc(onMouseMotion);
	glutReshapeFunc(onWindowReshape);
	
}
// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.529, 0.808, 0.922, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	gpuProgram.setUniform(camera.getPos(), "wEye");
	for (std::pair<const vec3i, Block> b : blocks)
		b.second.Draw();
	glutSwapBuffers(); // exchange buffers for double buffering
}

bool pressed[256] = { false };
int lastPressedNumber = 0;

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
	key = tolower(key);
	pressed[key] = true;
	if (key >= '0' && key <= '9')
		lastPressedNumber = key - '0';
	//prf((int)key, "key");
	if (key == 'c')
		captureMouse = !captureMouse;
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
	key = tolower(key);
	pressed[key] = false;
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	
	if (state == GLUT_DOWN) {
		Hit besthit;
		vec3 wLookDir = camera.wLookDir();
		for (std::pair<const vec3i, Block> b : blocks) {
			Hit h = b.second.intersect(camera.getPos(), wLookDir);
			if (h.t >= 0 && (besthit.t < 0 || besthit.t > h.t))
				besthit = h;
		}
        if (besthit.t < 0)
			return;
		if (button == GLUT_LEFT_BUTTON)
			deleteBlock(besthit.pos);
		else if (button == GLUT_RIGHT_BUTTON)
			placeBlock(besthit.pos + besthit.normal, lastPressedNumber);
    }
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	static long prev = 0;
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	long deltams = time - prev;
	prev = time;
	float delta = deltams / 1000.0f;
	camera.move(vec3(pressed[(int)'d'] - pressed[(int)'a'],
					 pressed[(int)' '] - (pressed[237/*hosszu i*/] || pressed[(int)'y']),
					 pressed[(int)'s'] - pressed[(int)'w']
					 ) * 60, delta);
	camera.rotate(vec2(pressed[(int)'j'] - pressed[(int)'l'], pressed[(int)'i'] - pressed[(int)'k']) * delta * 2);
	
	glutPostRedisplay();
}
