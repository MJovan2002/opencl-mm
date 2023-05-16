use std::ffi::CString;
use std::{any, array, mem, ptr};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::ops::{Add, Index, IndexMut, Mul};

use anyhow::anyhow;

use opencl3::types::{cl_device_id, cl_mem, cl_program, cl_ulong};
use opencl3::device::{cl_command_queue, cl_context, CL_DEVICE_TYPE_ALL, get_device_ids, release_device};
use opencl3::memory::{CL_MEM_READ_WRITE, create_buffer, release_mem_object};
use opencl3::program::{build_program, create_program_with_source, release_program};
use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CL_QUEUE_PROPERTIES, create_command_queue_with_properties, enqueue_nd_range_kernel, enqueue_read_buffer, enqueue_write_buffer, finish, release_command_queue};
use opencl3::context::context::{create_context_from_type, release_context};
use opencl3::platform::platform::get_platform_ids;
use opencl3::kernel::{create_kernel, release_kernel, set_kernel_arg};
use opencl3::event::{CL_PROFILING_COMMAND_END, CL_PROFILING_COMMAND_START, get_event_profiling_info, release_event};

pub fn scope<'env, F: for<'scope> FnOnce(&'scope Scope<'env, 'scope>) -> T, T>(f: F) -> anyhow::Result<T> {
    let scope = Scope::new()?;
    let t = f(&scope);
    unsafe {
        scope.mem.borrow_mut().drain(..).try_for_each(|mem| release_mem_object(mem)).map_err(|e| anyhow!("{e:?}"))?;
        release_program(scope.program).map_err(|e| anyhow!("{e:?}"))?;
        release_command_queue(scope.queue).map_err(|e| anyhow!("{e:?}"))?;
        release_context(scope.context).map_err(|e| anyhow!("{e:?}"))?;
        release_device(scope.device_id).map_err(|e| anyhow!("{e:?}"))?;
    }
    Ok(t)
}

pub struct Scope<'env: 'scope, 'scope> {
    device_id: cl_device_id,
    context: cl_context,
    queue: cl_command_queue,
    program: cl_program,
    mem: RefCell<Vec<cl_mem>>, // todo: remove refcell
    _marker0: PhantomData<&'env mut &'env ()>,
    _marker1: PhantomData<&'scope mut &'scope ()>,
}

impl<'env: 'scope, 'scope> Scope<'env, 'scope> {
    fn new() -> anyhow::Result<Self> {
        let platform_id = get_platform_ids().into_iter().flatten().next().ok_or(anyhow!("no_platform"))?;
        let device_id = get_device_ids(
            platform_id,
            CL_DEVICE_TYPE_ALL,
        ).into_iter().flatten().next().ok_or(anyhow!("no devices"))?;

        let context = create_context_from_type(
            CL_DEVICE_TYPE_ALL,
            ptr::null(),
            None,
            ptr::null_mut(),
        ).map_err(|e| anyhow!("{e:?}"))?;
        let queue = unsafe {
            create_command_queue_with_properties(
                context,
                device_id,
                [CL_QUEUE_PROPERTIES as _, CL_QUEUE_PROFILING_ENABLE, 0].as_ptr(),
            )
        }.map_err(|e| anyhow!("{e:?}"))?;
        let program = create_program_with_source(context, &[include_str!("../mul.cl")]).map_err(|e| anyhow!("{e:?}"))?;
        build_program(
            program,
            &[device_id],
            CString::new("").unwrap().as_c_str(),
            None,
            ptr::null_mut(),
        ).map_err(|e| anyhow!("{e:?}"))?;

        Ok(Self {
            device_id,
            context,
            queue,
            program,
            mem: Default::default(),
            _marker0: PhantomData,
            _marker1: PhantomData,
        })
    }

    fn get_mem(&self, size: usize) -> anyhow::Result<cl_mem> {
        let mem = unsafe {
            create_buffer(
                self.context,
                CL_MEM_READ_WRITE,
                size,
                ptr::null_mut(),
            )
        }.map_err(|e| anyhow!("{e:?}"))?;
        self.mem.borrow_mut().push(mem);
        Ok(mem)
    }

    pub fn create<const N: usize, const M: usize, T>(&'scope self) -> anyhow::Result<Matrix<'env, '_, N, M, T>> where T: Default {
        Ok(Matrix::new(self)?)
    }

    pub fn create_with<const N: usize, const M: usize, T>(&'scope self, data: [[T; M]; N]) -> anyhow::Result<Matrix<'env, '_, N, M, T>> {
        Ok(Matrix::with(data, self)?)
    }
}

pub struct Matrix<'env: 'scope, 'scope, const N: usize, const M: usize, T> {
    host_data: [[T; M]; N],
    guest_data: cl_mem,
    context: &'scope Scope<'env, 'scope>,
    _marker0: PhantomData<&'env mut &'env ()>,
    _marker1: PhantomData<&'scope mut &'scope ()>,
}

impl<'env: 'scope, 'scope, const N: usize, const M: usize, T> Matrix<'env, 'scope, N, M, T> {
    fn new(context: &'scope Scope<'env, 'scope>) -> anyhow::Result<Self> where T: Default {
        Self::with(array::from_fn(|_| array::from_fn(|_| T::default())), context)
    }

    fn with(data: [[T; M]; N], context: &'scope Scope<'env, 'scope>) -> anyhow::Result<Self> {
        Ok(Self {
            host_data: data,
            guest_data: context.get_mem(N * M * mem::size_of::<T>())?,
            context,
            _marker0: PhantomData,
            _marker1: PhantomData,
        })
    }

    unsafe fn write_back(&self) -> anyhow::Result<()> {
        enqueue_write_buffer(
            self.context.queue,
            self.guest_data,
            true.into(),
            0,
            N * M * mem::size_of::<T>(),
            self.host_data.as_ptr().cast(),
            0,
            ptr::null(),
        ).map_err(|e| anyhow!("{e:?}"))?;
        Ok(())
    }

    unsafe fn read_back(&mut self) -> anyhow::Result<()> {
        enqueue_read_buffer(
            self.context.queue,
            self.guest_data,
            true.into(),
            0,
            N * M * mem::size_of::<T>(),
            self.host_data.as_mut_ptr().cast(),
            0,
            ptr::null(),
        ).map_err(|e| anyhow!("{e:?}"))?;
        Ok(())
    }

    pub fn try_mul<const K: usize, F: FnMut(cl_ulong)>(&self, rhs: &Matrix<'env, 'scope, M, K, T>, mut f: F) -> anyhow::Result<Matrix<'env, 'scope, N, K, T>> where T: Default + MMul {
        unsafe {
            self.write_back()?;
            rhs.write_back()?;

            let mut o = Matrix::new(self.context)?;

            let kernel = create_kernel(self.context.program, &CString::new(format!("mul_{}", any::type_name::<T>())).unwrap()).map_err(|e| anyhow!("{e:?}"))?;

            set_kernel_arg(kernel, 0, mem::size_of::<i32>(), (&(N as i32) as *const i32).cast()).map_err(|e| anyhow!("{e:?}"))?;
            set_kernel_arg(kernel, 1, mem::size_of::<i32>(), (&(M as i32) as *const i32).cast()).map_err(|e| anyhow!("{e:?}"))?;
            set_kernel_arg(kernel, 2, mem::size_of::<i32>(), (&(K as i32) as *const i32).cast()).map_err(|e| anyhow!("{e:?}"))?;
            set_kernel_arg(kernel, 3, mem::size_of::<cl_mem>(), (&self.guest_data as *const cl_mem).cast()).map_err(|e| anyhow!("{e:?}"))?;
            set_kernel_arg(kernel, 4, mem::size_of::<cl_mem>(), (&rhs.guest_data as *const cl_mem).cast()).map_err(|e| anyhow!("{e:?}"))?;
            set_kernel_arg(kernel, 5, mem::size_of::<cl_mem>(), (&o.guest_data as *const cl_mem).cast()).map_err(|e| anyhow!("{e:?}"))?;

            let exec_event = enqueue_nd_range_kernel(
                self.context.queue,
                kernel,
                2,
                ptr::null(),
                [N, K].as_ptr(),
                [4, 4].as_ptr(),
                0,
                ptr::null(),
            ).map_err(|e| anyhow!("{e:?}"))?;
            o.read_back()?;

            finish(self.context.queue).map_err(|e| anyhow!("{e:?}"))?;

            let start = get_event_profiling_info(exec_event, CL_PROFILING_COMMAND_START).map_err(|e| anyhow!("{e:?}"))?.to_ulong();
            let end = get_event_profiling_info(exec_event, CL_PROFILING_COMMAND_END).map_err(|e| anyhow!("{e:?}"))?.to_ulong();
            f(end - start);

            release_event(exec_event).map_err(|e| anyhow!("{e:?}"))?;
            release_kernel(kernel).map_err(|e| anyhow!("{e:?}"))?;

            Ok(o)
        }
    }

    pub fn get(&self, i: usize, j: usize) -> Option<&T> {
        Some(self.host_data.get(i)?.get(j)?)
    }

    pub fn get_mut(&mut self, i: usize, j: usize) -> Option<&mut T> {
        Some(self.host_data.get_mut(i)?.get_mut(j)?)
    }

    pub fn host(&self) -> &[[T; M]; N] {
        &self.host_data
    }
}

impl<const N: usize, const M: usize, T> Index<(usize, usize)> for Matrix<'_, '_, N, M, T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        self.get(index.0, index.1).unwrap()
    }
}

impl<const N: usize, const M: usize, T> IndexMut<(usize, usize)> for Matrix<'_, '_, N, M, T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        self.get_mut(index.0, index.1).unwrap()
    }
}

impl<'env: 'scope, 'scope, const N: usize, const M: usize, const K: usize, T: Default + MMul + 'scope> Mul<Matrix<'env, 'scope, M, K, T>> for Matrix<'env, 'scope, N, M, T> {
    type Output = Matrix<'env, 'scope, N, K, T>;

    fn mul(self, rhs: Matrix<'env, 'scope, M, K, T>) -> Self::Output {
        self.try_mul(&rhs, |_| {}).unwrap()
    }
}

impl<'env: 'scope, 'scope, const N: usize, const M: usize, T: PartialEq> PartialEq for Matrix<'env, 'scope, N, M, T> {
    fn eq(&self, other: &Self) -> bool {
        self.host_data == other.host_data
    }
}

impl<'env: 'scope, 'scope, const N: usize, const M: usize, T: Eq> Eq for Matrix<'env, 'scope, N, M, T> {}

pub trait MMul: Mul<Self, Output=Self> + Add<Self, Output=Self> + Sized {}

impl MMul for i8 {}

impl MMul for i16 {}

impl MMul for i32 {}

impl MMul for i64 {}

impl MMul for u8 {}

impl MMul for u16 {}

impl MMul for u32 {}

impl MMul for u64 {}

impl MMul for f32 {}

impl MMul for f64 {}
