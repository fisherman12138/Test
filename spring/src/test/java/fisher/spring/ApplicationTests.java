package fisher.spring;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import fisher.spring.entity.User;
import fisher.spring.mapper.UserMapper;
import org.junit.jupiter.api.Test;
import org.junit.platform.commons.util.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.Arrays;
import java.util.List;

@SpringBootTest
class ApplicationTests {

    @Autowired
    private UserMapper userMapper;

    @Test
    public void test01(){
        List<User> users= userMapper.selectList(null);
        users.forEach(System.out::println);
    }

    /**
     * 查询ids
     * 查询id为1,2,3,5,6的用户
     */
    @Test
    public void test2(){
        userMapper.selectByIds(Arrays.asList(1,2,3,5,6)).forEach(System.out::println);
    }

    @Test
    public void test3(){
        System.out.println(userMapper.selectById(1));
    }

    @Test
    public void test4(){
        User user = new User();
        user.setName("老陈");
        user.setEmail("123321@qq.com");
        user.setAge(18);
        int insert = userMapper.insert(user);
        if (insert > 0 ){
            System.out.println("新增成功");
        }else{
            System.out.println("新增失败");
        }
    }

    /**
     * 修改userId为1的用户的邮箱
     */
    @Test
    public void test5(){
        User user = new User();
        user.setUserId(1L);
        user.setEmail("0000@qq.com");
        int i = userMapper.updateById(user);
        if (i > 0 ){
            System.out.println("修改成功");
        }else {
            System.out.println("修改失败");
        }
    }

    /**
     * 根据userId删除用户
     */
    @Test
    public void test6(){
        int i = userMapper.deleteById(7);
        if (i > 0){
            System.out.println("删除成功");
        }else {
            System.out.println("删除失败");
        }
    }


    @Test
    public void test7(){
        String email = "0000@qq.com";
        Integer age = 18;
        //创建条件构造器
        QueryWrapper<User> userQueryWrapper = new QueryWrapper<>();
        userQueryWrapper.like(StringUtils.isNotBlank(email), "email", email)
                .eq(age != null && age > 0, "age", age);
        userMapper.selectList(userQueryWrapper).forEach(System.out::println);
    }

    /**
     * 根据userId修改用户姓名, 邮箱
    */
    @Test
    public void test8(){
        String name = "老赵";
        String email = "4567879@qq.com";
        //更新条件构造器
        UpdateWrapper<User> userUpdateWrapper = new UpdateWrapper<>();
        //设置更新条件(需要修改的那一条数据)
        userUpdateWrapper.eq("user_id", 1);
        //设置需要修改的内容
        userUpdateWrapper.set(StringUtils.isNotBlank(name), "name", name)
                .set(StringUtils.isNotBlank(email), "email", email);
        //调用
        int update = userMapper.update(userUpdateWrapper);
        if (update > 0){
            System.out.println("修改成功");
        }else {
            System.out.println("修改失败");
        }
    }

    /**
     * 查询姓名中包含”肖”字样并且年龄大于等于18岁的用户。
     */
    @Test
    public void test9(){
        String name = "老";
        int age = 18;
        LambdaQueryWrapper<User> userLambdaQueryWrapper = new LambdaQueryWrapper<>();
        userLambdaQueryWrapper.like(StringUtils.isNotBlank(name), User::getName, name)
                .ge(User::getAge, age);
        userMapper.selectList(userLambdaQueryWrapper).forEach(System.out::println);
    }

    /**
     * 修改userId为1的用户的姓名,邮箱
     */
    @Test
    public void test10(){
        String name = "小赵";
        String email = "119@qq.com";
        LambdaUpdateWrapper<User> userLambdaUpdateWrapper = new LambdaUpdateWrapper<>();
        //设置更新条件-需要修改哪条数据
        userLambdaUpdateWrapper.eq(User::getUserId, 1);
        //设置更新内容
        userLambdaUpdateWrapper.set(StringUtils.isNotBlank(name), User::getName, name)
                .set(StringUtils.isNotBlank(email), User::getEmail, email);
        int update = userMapper.update(userLambdaUpdateWrapper);
        if (update > 0){
            System.out.println("修改成功");
        }else {
            System.out.println("修改失败");
        }
    }

    @Test
    public void test11(){
        int pageNo = 3;
        int pageSize = 2;
        //创建page
        Page<User> userPage = new Page<>(pageNo, pageSize);
        userMapper.selectPage(userPage, null);
        System.out.println("数据: " + userPage.getRecords());
        System.out.println("总条数: " + userPage.getTotal());
    }

    /**
     * 自定义分页查询
     */
    @Test
    public void test12(){
        int age = 18;
        int pageNo = 1;
        int pageSize = 2;
        Page<User> userPage = new Page<>(pageNo, pageSize);
        userMapper.selectPage(userPage, age);
        System.out.println("数据: " + userPage.getRecords());
        System.out.println("总条数: " + userPage.getTotal());
    }


}
