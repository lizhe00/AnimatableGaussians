#ifndef _VECTOR_OPERATION_HPP_
#define _VECTOR_OPERATION_HPP_

#include <vector_functions.h>
#include <stdio.h>

// overloading
__host__ __device__ __forceinline__ float3
make_float3(const float4 &_v)
{
	return make_float3(_v.x, _v.y, _v.z);
}

__host__ __device__ __forceinline__ float4
make_float4(const float3 &_v, const float &_w)
{
	return make_float4(_v.x, _v.y, _v.z, _w);
}

__host__ __device__ __forceinline__ float
dot(const float3& v1, const float3& v2)
{
	return v1.x * v2.x + v1.y*v2.y + v1.z*v2.z;
}

__host__ __device__ __forceinline__ float3&
operator+=(float3& vec, const float& v)
{
	vec.x += v;  vec.y += v;  vec.z += v; return vec;
}

__host__ __device__ __forceinline__ float3
operator+(const float3& v1, const float3& v2)
{
	return make_float3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

__host__ __device__ __forceinline__ float3&
operator*=(float3& vec, const float& v)
{
	vec.x *= v;  vec.y *= v;  vec.z *= v; return vec;
}

__host__ __device__ __forceinline__ float4
operator+(const float4& v1, const float4& v2)
{
    return make_float4(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w);
}

__host__ __device__ __forceinline__ float3
operator-(const float3& v1, const float3& v2)
{
	return make_float3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

__host__ __device__ __forceinline__ float4
operator-(const float4& v1, const float4& v2)
{
	return make_float4(v1.x-v2.x, v1.y-v2.y, v1.z-v2.z, v1.w-v2.w);
}

__host__ __device__ __forceinline__ float3
operator*(const float& v, const float3& v1)
{
	return make_float3(v * v1.x, v * v1.y, v * v1.z);
}

__host__ __device__ __forceinline__ float3
operator*(const float3& v1, const float& v)
{
	return make_float3(v1.x * v, v1.y * v, v1.z * v);
}

__host__ __device__ __forceinline__ float
fabs_sum(const float3 &v)
{
	return fabsf(v.x) + fabsf(v.y) + fabsf(v.z);
}

__host__ __device__ __forceinline__ float
squared_norm(const float3 &v)
{
	return dot(v, v);
}

__host__ __device__ __forceinline__ float
squared_norm(const float2 &v)
{
	return v.x*v.x + v.y*v.y;
}

__host__ __device__ __forceinline__ float
norm(const float3& v)
{
	return sqrt(dot(v, v));
}

__host__ __device__ __forceinline__ float3
normalized(const float3 &v)
{
	return v * (1.0f/sqrtf(dot(v, v)));
}

__host__ __device__ __forceinline__ float3
cross(const float3& v1, const float3& v2)
{
	return make_float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

__host__ __device__ __forceinline__ float
dot(const float4 &v1, const float4 &v2)
{
	return v1.w*v2.w+v1.x*v2.x+v1.y*v2.y+v1.z*v2.z;
}

__host__ __device__ __forceinline__ float
squared_norm(const float4 &v)
{
	return dot(v, v);
}

__host__ __device__ __forceinline__ float
norm(const float4 &v4)
{
	return sqrtf(squared_norm(v4));
}

__host__ __device__ __forceinline__ float4 operator*(const float4 &v, float s)
{
	return make_float4(v.x*s, v.y*s, v.z*s, v.w*s);
}

__host__ __device__ __forceinline__ float4 operator*(float s, const float4 &v)
{
	return make_float4(s*v.x, s*v.y, s*v.z, s*v.w);
}

__host__ __device__ __forceinline__ float4
normalized(const float4 &v)
{
	return v * (1.0f/sqrtf(dot(v, v)));
}
#endif

__host__ __device__ __forceinline__ float3
operator-(const float3 &v)
{
	return make_float3(-v.x, -v.y, -v.z);
}

__host__ __device__ __forceinline__ float4
operator-(const float4 &v)
{
	return make_float4(-v.x, -v.y, -v.z, -v.w);
}

struct mat33 {
	__host__ __device__ mat33() {}
	__host__ __device__ mat33(const float3 &_a0, const float3 &_a1, const float3 &_a2) { cols[0] = _a0; cols[1] = _a1; cols[2] = _a2; }
	__host__ __device__ mat33(const float *_data)
	{
		/*_data MUST have at least 9 float elements, ctor does not check range*/
		cols[0] = make_float3(_data[0], _data[1], _data[2]);
		cols[1] = make_float3(_data[3], _data[4], _data[5]);
		cols[2] = make_float3(_data[6], _data[7], _data[8]);
	}

	__host__ __device__ const float& m00() const { return cols[0].x; }
	__host__ __device__ const float& m10() const { return cols[0].y; }
	__host__ __device__ const float& m20() const { return cols[0].z; }
	__host__ __device__ const float& m01() const { return cols[1].x; }
	__host__ __device__ const float& m11() const { return cols[1].y; }
	__host__ __device__ const float& m21() const { return cols[1].z; }
	__host__ __device__ const float& m02() const { return cols[2].x; }
	__host__ __device__ const float& m12() const { return cols[2].y; }
	__host__ __device__ const float& m22() const { return cols[2].z; }

	__host__ __device__ float& m00() { return cols[0].x; }
	__host__ __device__ float& m10() { return cols[0].y; }
	__host__ __device__ float& m20() { return cols[0].z; }
	__host__ __device__ float& m01() { return cols[1].x; }
	__host__ __device__ float& m11() { return cols[1].y; }
	__host__ __device__ float& m21() { return cols[1].z; }
	__host__ __device__ float& m02() { return cols[2].x; }
	__host__ __device__ float& m12() { return cols[2].y; }
	__host__ __device__ float& m22() { return cols[2].z; }

	__host__ __device__ mat33 transpose() const
	{
		float3 row0 = make_float3(cols[0].x, cols[1].x, cols[2].x);
		float3 row1 = make_float3(cols[0].y, cols[1].y, cols[2].y);
		float3 row2 = make_float3(cols[0].z, cols[1].z, cols[2].z);
		return mat33(row0, row1, row2);
	}

	__host__ __device__ mat33 operator* (const mat33 &_mat) const
	{
		mat33 mat;
		mat.m00() = m00()*_mat.m00() + m01()*_mat.m10() + m02()*_mat.m20();
		mat.m01() = m00()*_mat.m01() + m01()*_mat.m11() + m02()*_mat.m21();
		mat.m02() = m00()*_mat.m02() + m01()*_mat.m12() + m02()*_mat.m22();
		mat.m10() = m10()*_mat.m00() + m11()*_mat.m10() + m12()*_mat.m20();
		mat.m11() = m10()*_mat.m01() + m11()*_mat.m11() + m12()*_mat.m21();
		mat.m12() = m10()*_mat.m02() + m11()*_mat.m12() + m12()*_mat.m22();
		mat.m20() = m20()*_mat.m00() + m21()*_mat.m10() + m22()*_mat.m20();
		mat.m21() = m20()*_mat.m01() + m21()*_mat.m11() + m22()*_mat.m21();
		mat.m22() = m20()*_mat.m02() + m21()*_mat.m12() + m22()*_mat.m22();
		return mat;
	}

	__host__ __device__ mat33 operator+ (const mat33 &_mat) const
	{
		mat33 mat_sum;
		mat_sum.m00() = m00() + _mat.m00();
		mat_sum.m01() = m01() + _mat.m01();
		mat_sum.m02() = m02() + _mat.m02();

		mat_sum.m10() = m10() + _mat.m10();
		mat_sum.m11() = m11() + _mat.m11();
		mat_sum.m12() = m12() + _mat.m12();

		mat_sum.m20() = m20() + _mat.m20();
		mat_sum.m21() = m21() + _mat.m21();
		mat_sum.m22() = m22() + _mat.m22();

		return mat_sum;
	}

	__host__ __device__ mat33 operator- (const mat33 &_mat) const
	{
		mat33 mat_diff;
		mat_diff.m00() = m00() - _mat.m00();
		mat_diff.m01() = m01() - _mat.m01();
		mat_diff.m02() = m02() - _mat.m02();

		mat_diff.m10() = m10() - _mat.m10();
		mat_diff.m11() = m11() - _mat.m11();
		mat_diff.m12() = m12() - _mat.m12();

		mat_diff.m20() = m20() - _mat.m20();
		mat_diff.m21() = m21() - _mat.m21();
		mat_diff.m22() = m22() - _mat.m22();

		return mat_diff;			
	}

	__host__ __device__ mat33 operator-() const
	{
		mat33 mat_neg;
		mat_neg.m00() = -m00();
		mat_neg.m01() = -m01();
		mat_neg.m02() = -m02();

		mat_neg.m10() = -m10();
		mat_neg.m11() = -m11();
		mat_neg.m12() = -m12();

		mat_neg.m20() = -m20();
		mat_neg.m21() = -m21();
		mat_neg.m22() = -m22();

		return mat_neg;
	}

	__host__ __device__ mat33& operator*= (const mat33 &_mat)
	{
		*this = *this * _mat;
		return *this;
	}

	__host__ __device__ float3 operator* (const float3 &_vec) const
	{
		float x = m00()*_vec.x + m01()*_vec.y + m02()*_vec.z;
		float y = m10()*_vec.x + m11()*_vec.y + m12()*_vec.z;
		float z = m20()*_vec.x + m21()*_vec.y + m22()*_vec.z;
		return make_float3(x, y, z);
	}

    __host__ __device__ mat33 operator* (const float &_f) const 
    {
        return mat33(cols[0] * _f, cols[1] * _f, cols[2] * _f);
    }

    __host__ __device__ mat33 inverse() const 
    {
        /*
        Reference: 
        d = (m00_*m11_*m22_ - m00_*m12_*m21_ - m01_*m10_*m22_ + m01_*m12_*m20_ + m02_*m10_*m21_ - m02_*m11_*m20_)
        ans =

        [  (m11_*m22_ - m12_*m21_)/d, -(m01_*m22_ - m02_*m21_)/d,  (m01_*m12_ - m02_*m11_)/d]
        [ -(m10_*m22_ - m12_*m20_)/d,  (m00_*m22_ - m02_*m20_)/d, -(m00_*m12_ - m02_*m10_)/d]
        [  (m10_*m21_ - m11_*m20_)/d, -(m00_*m21_ - m01_*m20_)/d,  (m00_*m11_ - m01_*m10_)/d]

        */

        float d = m00()*m11()*m22() - m00()*m12()*m21() - m01()*m10()*m22() + m01()*m12()*m20() + m02()*m10()*m21() - m02()*m11()*m20();
        mat33 r;
        r.m00() = (m11()*m22() - m12()*m21());
        r.m01() = -(m01()*m22() - m02()*m21());
        r.m02() = (m01()*m12() - m02()*m11());
        r.m10() = -(m10()*m22() - m12()*m20());
        r.m11() = (m00()*m22() - m02()*m20());
        r.m12() = -(m00()*m12() - m02()*m10());
        r.m20() = (m10()*m21() - m11()*m20());
        r.m21() = -(m00()*m21() - m01()*m20());
        r.m22() = (m00()*m11() - m01()*m10());

        return r * (1.f / d);
    }

	__host__ __device__ void set_identity()
	{
		cols[0] = make_float3(1, 0, 0);
		cols[1] = make_float3(0, 1, 0);
		cols[2] = make_float3(0, 0, 1);
	}

	__host__ __device__ static mat33 identity()
	{
		mat33 idmat;
		idmat.set_identity();
		return idmat;
	}

    __host__ __device__ static mat33 zero()
    {
        mat33 idmat;
        idmat.cols[0] = make_float3(0, 0, 0);
        idmat.cols[1] = make_float3(0, 0, 0);
        idmat.cols[2] = make_float3(0, 0, 0);
        return idmat;
    }

	__host__ __device__ void print() const
	{
		printf("%f %f %f \n", m00(), m01(), m02());
		printf("%f %f %f \n", m10(), m11(), m12());
		printf("%f %f %f \n", m20(), m21(), m22());
	}

	float3 cols[3]; /*colume major*/
};

/*rotation and translation*/
struct mat34 {
	__host__ __device__ mat34() {}
	__host__ __device__ mat34(const mat33 &_rot, const float3 &_trans) : rot(_rot), trans(_trans) {}

	__host__ __device__ static mat34 identity()
	{
		return mat34(mat33::identity(), make_float3(0, 0, 0));
	}

    __host__ __device__ static mat34 zeros()
    {
        return mat34(mat33::zero(), make_float3(0, 0, 0));
    }

	__host__ __device__ mat34 operator* (const mat34 &_right_se3) const
	{
		mat34 se3;
		se3.rot = rot*_right_se3.rot;
		se3.trans = rot*_right_se3.trans + trans;
		return se3;
	}

	__host__ __device__ mat34 operator* (const float &_w) const
	{
		mat34 se3;
		se3.rot = rot;
		se3.trans = trans;

		//se3.rot.m00() *= _w*se3.rot.m00();	se3.rot.m01() *= _w*se3.rot.m01();	se3.rot.m02() *= _w*se3.rot.m02();
		//se3.rot.m10() *= _w*se3.rot.m10();	se3.rot.m11() *= _w*se3.rot.m11();	se3.rot.m12() *= _w*se3.rot.m12();
		//se3.rot.m20() *= _w*se3.rot.m20();	se3.rot.m21() *= _w*se3.rot.m21();	se3.rot.m22() *= _w*se3.rot.m22();

		//se3.trans.x *= _w*se3.trans.x;	se3.trans.y *= _w*se3.trans.y;	se3.trans.z *= _w*se3.trans.z;


		se3.rot.m00() *= _w;	se3.rot.m01() *= _w;	se3.rot.m02() *= _w;
		se3.rot.m10() *= _w;	se3.rot.m11() *= _w;	se3.rot.m12() *= _w;
		se3.rot.m20() *= _w;	se3.rot.m21() *= _w;	se3.rot.m22() *= _w;

		se3.trans.x *= _w;	se3.trans.y *= _w;	se3.trans.z *= _w;
		return se3;
	}

	__host__ __device__ mat34 operator+ (const mat34 &_T) const
	{
		mat34 se3;
		se3.rot = rot;
		se3.trans = trans;

		se3.rot.m00() += _T.rot.m00();	se3.rot.m01() += _T.rot.m01();	se3.rot.m02() += _T.rot.m02();
		se3.rot.m10() += _T.rot.m10();	se3.rot.m11() += _T.rot.m11();	se3.rot.m12() += _T.rot.m12();
		se3.rot.m20() += _T.rot.m20();	se3.rot.m21() += _T.rot.m21();	se3.rot.m22() += _T.rot.m22();

		se3.trans.x += _T.trans.x;	se3.trans.y += _T.trans.y;	se3.trans.z += _T.trans.z;

		return se3;
	}


	__host__ __device__ mat34& operator*= (const mat34 &_right_se3)
	{
		*this = *this * _right_se3;
		return *this;
	}

    __host__ __device__ float3 operator* (const float3 &_vec) const
    {
        return rot * _vec + trans;
    }

    __host__ __device__ float4 operator* (const float4 &_vec) const
    {
        float3 v =  rot * make_float3(_vec) + _vec.w * trans;
        return make_float4(v, _vec.w);
    }

    __host__ __device__ mat34 inverse() const
    {
        mat34 r;
		mat33 rot_inv = rot.inverse();
        r.rot = rot_inv;
        r.trans = -rot_inv * trans;
        return r;
    }

	__host__ __device__ void print() const
	{
		printf("%f %f %f %f \n", rot.m00(), rot.m01(), rot.m02(), trans.x);
		printf("%f %f %f %f \n", rot.m10(), rot.m11(), rot.m12(), trans.y);
		printf("%f %f %f %f \n", rot.m20(), rot.m21(), rot.m22(), trans.z);
	}

	bool checkOrthogonalityMat34()
	{
		// orthogonality check
		mat33 R = rot;
		mat33 RTR = R.transpose() * R;

		bool ortho = true;

		if (fabsf(RTR.m00() - 1.0f) > 1e-6f) ortho = false;
		if (fabsf(RTR.m01() - 0.0f) > 1e-6f) ortho = false;
		if (fabsf(RTR.m02() - 0.0f) > 1e-6f) ortho = false;

		if (fabsf(RTR.m10() - 0.0f) > 1e-6f) ortho = false;
		if (fabsf(RTR.m11() - 1.0f) > 1e-6f) ortho = false;
		if (fabsf(RTR.m12() - 0.0f) > 1e-6f) ortho = false;

		if (fabsf(RTR.m20() - 0.0f) > 1e-6f) ortho = false;
		if (fabsf(RTR.m21() - 0.0f) > 1e-6f) ortho = false;
		if (fabsf(RTR.m22() - 1.0f) > 1e-6f) ortho = false;

		float det =
			RTR.m00() * RTR.m11() * RTR.m22() +
			RTR.m01() * RTR.m12() * RTR.m20() +
			RTR.m02() * RTR.m10() * RTR.m21() -
			RTR.m02() * RTR.m11() * RTR.m20() -
			RTR.m01() * RTR.m10() * RTR.m22() -
			RTR.m00() * RTR.m12() * RTR.m21();

		if (fabsf(det - 1.0f) > 1e-6f) ortho = false;

		return ortho;
	}

	mat33 rot;
	float3 trans;
};

/*outer production of two float3*/
__host__ __device__ __forceinline__ mat33
outer_prod(const float3 &v0, const float3 &v1)
{
	return mat33(v0*v1.x, v0*v1.y, v0*v1.z);
}

// symmetric inverse of mat33
__host__ __device__ __forceinline__
mat33 sym_inv(const mat33 &_A)
{
	float det = _A.m00()*_A.m11()*_A.m22() +
			  2*_A.m01()*_A.m02()*_A.m12() -
			   (_A.m00()*_A.m12()*_A.m12() +
			    _A.m11()*_A.m02()*_A.m02() +
				_A.m22()*_A.m01()*_A.m01());

	mat33 A_inv;

	if (fabs(det) < 1e-10f) {
		A_inv.set_identity();
	}
	else {
		float det_inv = 1/det;
		A_inv.m00() = det_inv*(_A.m11()*_A.m22() - _A.m12()*_A.m12());
		A_inv.m11() = det_inv*(_A.m00()*_A.m22() - _A.m02()*_A.m02());
		A_inv.m22() = det_inv*(_A.m00()*_A.m11() - _A.m01()*_A.m01());
		A_inv.m01() = A_inv.m10() = det_inv*(_A.m02()*_A.m12() - _A.m01()*_A.m22());
		A_inv.m02() = A_inv.m20() = det_inv*(_A.m01()*_A.m12() - _A.m02()*_A.m11());
		A_inv.m12() = A_inv.m21() = det_inv*(_A.m02()*_A.m01() - _A.m00()*_A.m12());
	}

	return A_inv;
}